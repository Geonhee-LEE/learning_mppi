#!/usr/bin/env python3
"""
MPPI Controller ROS2 Node

ROS2 wrapper for MPPI controllers supporting all 9 variants.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from typing import Optional

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)
from mppi_controller.controllers.mppi.base_mppi import MPPIController
from mppi_controller.controllers.mppi.tube_mppi import TubeMPPIController
from mppi_controller.controllers.mppi.log_mppi import LogMPPIController
from mppi_controller.controllers.mppi.tsallis_mppi import TsallisMPPIController
from mppi_controller.controllers.mppi.risk_aware_mppi import RiskAwareMPPIController
from mppi_controller.controllers.mppi.smooth_mppi import SmoothMPPIController
from mppi_controller.controllers.mppi.stein_variational_mppi import (
    SteinVariationalMPPIController,
)
from mppi_controller.controllers.mppi.spline_mppi import SplineMPPIController
from mppi_controller.controllers.mppi.svg_mppi import SVGMPPIController
from mppi_controller.controllers.mppi.cbf_mppi import CBFMPPIController
from mppi_controller.controllers.mppi.shield_mppi import ShieldMPPIController
from mppi_controller.controllers.mppi.adaptive_shield_mppi import (
    AdaptiveShieldMPPIController,
)
from mppi_controller.controllers.mppi.adaptive_shield_mppi import AdaptiveShieldParams
from mppi_controller.controllers.mppi.shield_svg_mppi import (
    ShieldSVGMPPIController, ShieldSVGMPPIParams,
)
from mppi_controller.controllers.mppi.adaptive_shield_svg_mppi import (
    AdaptiveShieldSVGMPPIController, AdaptiveShieldSVGMPPIParams,
)
from mppi_controller.controllers.mppi.mppi_params import (
    MPPIParams,
    TubeMPPIParams,
    LogMPPIParams,
    TsallisMPPIParams,
    RiskAwareMPPIParams,
    SmoothMPPIParams,
    SteinVariationalMPPIParams,
    SplineMPPIParams,
    SVGMPPIParams,
    CBFMPPIParams,
    ShieldMPPIParams,
)
from mppi_controller.perception.obstacle_detector import ObstacleDetector
from mppi_controller.perception.obstacle_tracker import ObstacleTracker


class MPPIControllerNode(Node):
    """
    ROS2 Node for MPPI Control

    Subscribes:
        /odom (nav_msgs/Odometry): Robot odometry
        /reference_path (nav_msgs/Path): Reference trajectory

    Publishes:
        /cmd_vel (geometry_msgs/Twist): Control commands
        /mppi/info (custom): MPPI debug info
    """

    def __init__(self):
        super().__init__('mppi_controller_node')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self._get_parameters()

        # Create model
        self.model = self._create_model()

        # Create controller
        self.controller = self._create_controller()

        # State
        self.current_state: Optional[np.ndarray] = None
        self.reference_path: Optional[Path] = None
        self.last_control_time = self.get_clock().now()

        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile
        )

        self.path_sub = self.create_subscription(
            Path,
            '/reference_path',
            self.path_callback,
            qos_profile
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile
        )

        # LaserScan obstacle detection
        self.obstacle_detector = None
        self.obstacle_tracker = None
        if self.get_parameter('scan_enabled').value:
            self.obstacle_detector = ObstacleDetector()
            self.obstacle_tracker = ObstacleTracker()
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, 10
            )
            self.get_logger().info('  LaserScan obstacle detection enabled')

        # Control timer
        control_rate = self.get_parameter('control_rate').value
        self.control_timer = self.create_timer(
            1.0 / control_rate,
            self.control_loop
        )

        self.get_logger().info(f'MPPI Controller Node initialized')
        self.get_logger().info(f'  Model: {self.get_parameter("model_type").value}')
        self.get_logger().info(f'  Controller: {self.get_parameter("controller_type").value}')
        self.get_logger().info(f'  Control rate: {control_rate} Hz')

    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        # Model parameters
        self.declare_parameter('model_type', 'kinematic')  # kinematic, dynamic
        self.declare_parameter('v_max', 1.0)
        self.declare_parameter('omega_max', 1.0)

        # Controller parameters
        self.declare_parameter('controller_type', 'vanilla')  # vanilla, tube, log, etc.
        self.declare_parameter('N', 30)  # horizon
        self.declare_parameter('dt', 0.05)  # timestep
        self.declare_parameter('K', 1024)  # samples
        self.declare_parameter('lambda_', 1.0)  # temperature
        self.declare_parameter('sigma', [0.5, 0.5])  # noise std

        # Cost weights
        self.declare_parameter('Q', [10.0, 10.0, 1.0])  # state tracking
        self.declare_parameter('R', [0.1, 0.1])  # control effort
        self.declare_parameter('Qf', [20.0, 20.0, 2.0])  # terminal

        # CBF/Shield parameters
        self.declare_parameter('cbf_alpha', 0.3)
        self.declare_parameter('cbf_weight', 1000.0)
        self.declare_parameter('cbf_safety_margin', 0.1)
        self.declare_parameter('cbf_use_safety_filter', False)
        self.declare_parameter('shield_enabled', True)

        # LaserScan obstacle detection
        self.declare_parameter('scan_enabled', False)

        # Node parameters
        self.declare_parameter('control_rate', 10.0)  # Hz
        self.declare_parameter('frame_id', 'map')

    def _get_parameters(self):
        """Get ROS2 parameters"""
        self.model_type = self.get_parameter('model_type').value
        self.controller_type = self.get_parameter('controller_type').value
        self.frame_id = self.get_parameter('frame_id').value

    def _create_model(self):
        """Create robot model based on parameters"""
        model_type = self.get_parameter('model_type').value
        v_max = self.get_parameter('v_max').value
        omega_max = self.get_parameter('omega_max').value

        if model_type == 'kinematic':
            return DifferentialDriveKinematic(v_max=v_max, omega_max=omega_max)
        elif model_type == 'dynamic':
            return DifferentialDriveDynamic(
                v_max=v_max,
                omega_max=omega_max,
                mass=10.0,
                inertia=1.0,
                c_v=0.1,
                c_omega=0.1
            )
        else:
            self.get_logger().error(f'Unknown model type: {model_type}')
            return DifferentialDriveKinematic(v_max=v_max, omega_max=omega_max)

    def _create_controller(self):
        """Create MPPI controller based on parameters"""
        # Get common parameters
        params_dict = {
            'N': self.get_parameter('N').value,
            'dt': self.get_parameter('dt').value,
            'K': self.get_parameter('K').value,
            'lambda_': self.get_parameter('lambda_').value,
            'sigma': np.array(self.get_parameter('sigma').value),
            'Q': np.array(self.get_parameter('Q').value),
            'R': np.array(self.get_parameter('R').value),
            'Qf': np.array(self.get_parameter('Qf').value),
        }

        controller_type = self.get_parameter('controller_type').value

        # Create controller based on type
        if controller_type == 'vanilla':
            params = MPPIParams(**params_dict)
            return MPPIController(self.model, params)

        elif controller_type == 'tube':
            params = TubeMPPIParams(
                **params_dict,
                tube_enabled=True,
                K_fb=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
                tube_margin=0.3,
            )
            return TubeMPPIController(self.model, params)

        elif controller_type == 'log':
            params = LogMPPIParams(**params_dict, use_baseline=True)
            return LogMPPIController(self.model, params)

        elif controller_type == 'tsallis':
            params = TsallisMPPIParams(**params_dict, tsallis_q=1.2)
            return TsallisMPPIController(self.model, params)

        elif controller_type == 'risk_aware':
            params = RiskAwareMPPIParams(**params_dict, cvar_alpha=0.7)
            return RiskAwareMPPIController(self.model, params)

        elif controller_type == 'smooth':
            params = SmoothMPPIParams(**params_dict, jerk_weight=1.0)
            return SmoothMPPIController(self.model, params)

        elif controller_type == 'svmpc':
            params = SteinVariationalMPPIParams(
                **params_dict,
                svgd_num_iterations=3,
                svgd_step_size=0.01,
            )
            return SteinVariationalMPPIController(self.model, params)

        elif controller_type == 'spline':
            params = SplineMPPIParams(
                **params_dict,
                spline_num_knots=8,
                spline_degree=3,
            )
            return SplineMPPIController(self.model, params)

        elif controller_type == 'svg':
            params = SVGMPPIParams(
                **params_dict,
                svg_num_guide_particles=32,
                svgd_num_iterations=3,
                svg_guide_step_size=0.01,
            )
            return SVGMPPIController(self.model, params)

        elif controller_type == 'cbf':
            params = CBFMPPIParams(
                **params_dict,
                cbf_obstacles=[],
                cbf_alpha=self.get_parameter('cbf_alpha').value,
                cbf_weight=self.get_parameter('cbf_weight').value,
                cbf_safety_margin=self.get_parameter('cbf_safety_margin').value,
                cbf_use_safety_filter=self.get_parameter('cbf_use_safety_filter').value,
            )
            return CBFMPPIController(self.model, params)

        elif controller_type == 'shield':
            params = ShieldMPPIParams(
                **params_dict,
                cbf_obstacles=[],
                cbf_alpha=self.get_parameter('cbf_alpha').value,
                cbf_weight=self.get_parameter('cbf_weight').value,
                cbf_safety_margin=self.get_parameter('cbf_safety_margin').value,
                cbf_use_safety_filter=self.get_parameter('cbf_use_safety_filter').value,
                shield_enabled=self.get_parameter('shield_enabled').value,
            )
            return ShieldMPPIController(self.model, params)

        elif controller_type == 'adaptive_shield':
            params = AdaptiveShieldParams(
                **params_dict,
                cbf_obstacles=[],
                cbf_alpha=self.get_parameter('cbf_alpha').value,
                cbf_weight=self.get_parameter('cbf_weight').value,
                cbf_safety_margin=self.get_parameter('cbf_safety_margin').value,
                cbf_use_safety_filter=self.get_parameter('cbf_use_safety_filter').value,
                shield_enabled=self.get_parameter('shield_enabled').value,
            )
            return AdaptiveShieldMPPIController(self.model, params)

        elif controller_type == 'adaptive_shield_svg':
            params = AdaptiveShieldSVGMPPIParams(
                **params_dict,
                cbf_obstacles=[],
                cbf_safety_margin=self.get_parameter('cbf_safety_margin').value,
                svgd_num_iterations=3, svgd_step_size=0.01,
                svg_num_guide_particles=32, svg_guide_step_size=0.01,
                shield_enabled=self.get_parameter('shield_enabled').value,
                shield_cbf_alpha=self.get_parameter('cbf_alpha').value,
            )
            return AdaptiveShieldSVGMPPIController(self.model, params)

        else:
            self.get_logger().warn(
                f'Unknown controller type: {controller_type}, using vanilla'
            )
            params = MPPIParams(**params_dict)
            return MPPIController(self.model, params)

    def odom_callback(self, msg: Odometry):
        """Odometry callback"""
        # Extract state [x, y, theta]
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Convert quaternion to yaw
        quat = msg.pose.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)

        if self.model_type == 'kinematic':
            self.current_state = np.array([x, y, theta])
        elif self.model_type == 'dynamic':
            # Extract velocities for dynamic model
            v = msg.twist.twist.linear.x
            omega = msg.twist.twist.angular.z
            self.current_state = np.array([x, y, theta, v, omega])

    def path_callback(self, msg: Path):
        """Reference path callback"""
        self.reference_path = msg

    def _path_to_reference_trajectory(self, path: Path) -> np.ndarray:
        """Convert ROS Path to reference trajectory array"""
        N = self.get_parameter('N').value
        state_dim = self.model.state_dim

        # Extract poses from path
        if len(path.poses) == 0:
            return np.zeros((N + 1, state_dim))

        # Convert poses to numpy array
        reference = np.zeros((min(len(path.poses), N + 1), state_dim))

        for i, pose_stamped in enumerate(path.poses[:N + 1]):
            pose = pose_stamped.pose
            x = pose.position.x
            y = pose.position.y

            # Convert quaternion to yaw
            quat = pose.orientation
            siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
            cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
            theta = np.arctan2(siny_cosp, cosy_cosp)

            if self.model_type == 'kinematic':
                reference[i] = [x, y, theta]
            elif self.model_type == 'dynamic':
                reference[i, :3] = [x, y, theta]
                # Dynamic model: velocities are zero in reference (for now)

        # Pad if needed
        if len(reference) < N + 1:
            last_pose = reference[-1]
            reference = np.vstack([
                reference,
                np.tile(last_pose, (N + 1 - len(reference), 1))
            ])

        return reference

    def control_loop(self):
        """Main control loop"""
        if self.current_state is None:
            self.get_logger().warn('No odometry received yet', throttle_duration_sec=5.0)
            return

        if self.reference_path is None:
            self.get_logger().warn('No reference path received yet', throttle_duration_sec=5.0)
            # Publish zero velocity
            self.publish_zero_velocity()
            return

        # Convert path to reference trajectory
        reference_trajectory = self._path_to_reference_trajectory(self.reference_path)

        # Compute control
        try:
            import time
            t_start = time.time()
            control, info = self.controller.compute_control(
                self.current_state,
                reference_trajectory
            )
            solve_time = (time.time() - t_start) * 1000.0  # ms

            # Publish control
            self.publish_control(control)

            # Log periodically
            if self.get_clock().now().nanoseconds % 1e9 < 1e8:  # ~every second
                self.get_logger().info(
                    f'Control: v={control[0]:.3f}, ω={control[1]:.3f}, '
                    f'Solve time: {solve_time:.2f}ms'
                )

        except Exception as e:
            self.get_logger().error(f'Control computation failed: {e}')
            self.publish_zero_velocity()

    def scan_callback(self, msg: LaserScan):
        """LaserScan callback — detect and track obstacles"""
        if self.obstacle_detector is None or self.obstacle_tracker is None:
            return

        robot_pose = self.current_state[:3] if self.current_state is not None else None

        detections = self.obstacle_detector.detect(
            np.array(msg.ranges),
            msg.angle_min,
            msg.angle_increment,
            robot_pose,
        )

        control_rate = self.get_parameter('control_rate').value
        dt = 1.0 / control_rate
        self.obstacle_tracker.update(detections, dt)

        if hasattr(self.controller, 'update_obstacles'):
            obstacles = self.obstacle_tracker.get_obstacles_as_tuples()
            self.controller.update_obstacles(obstacles)

    def publish_control(self, control: np.ndarray):
        """Publish control command"""
        msg = Twist()
        msg.linear.x = float(control[0])
        msg.angular.z = float(control[1])
        self.cmd_vel_pub.publish(msg)

    def publish_zero_velocity(self):
        """Publish zero velocity"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPPIControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
