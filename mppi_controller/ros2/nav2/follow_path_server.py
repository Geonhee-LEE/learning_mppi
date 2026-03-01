#!/usr/bin/env python3
"""MPPI FollowPath Action Server for Nav2 integration.

Provides a FollowPath action server compatible with nav2's BT Navigator,
replacing the C++ controller_server plugin with a Python action server.

Architecture:
    Nav2 BT Navigator → FollowPath action goal (nav_msgs/Path)
                       → MPPIFollowPathServer (LifecycleNode + ActionServer)
                       → /cmd_vel, /mppi/local_plan, feedback
"""

import time
import numpy as np
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

import tf2_ros

from mppi_controller.ros2.nav2.path_windower import PathWindower
from mppi_controller.ros2.nav2.costmap_converter import CostmapConverter
from mppi_controller.ros2.nav2.goal_checker import GoalChecker
from mppi_controller.ros2.nav2.progress_checker import ProgressChecker
from mppi_controller.ros2.nav2.tf2_helpers import (
    get_robot_state_from_tf,
    transform_path_to_frame,
    quaternion_to_yaw,
    pose_stamped_to_array,
)

# Models
from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)
from mppi_controller.models.dynamic.differential_drive_dynamic import (
    DifferentialDriveDynamic,
)

# Controllers
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
from mppi_controller.controllers.mppi.adaptive_shield_svg_mppi import (
    AdaptiveShieldSVGMPPIController,
)
from mppi_controller.controllers.mppi.shield_svg_mppi import ShieldSVGMPPIController
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
from mppi_controller.controllers.mppi.adaptive_shield_mppi import AdaptiveShieldParams
from mppi_controller.controllers.mppi.shield_svg_mppi import ShieldSVGMPPIParams
from mppi_controller.controllers.mppi.adaptive_shield_svg_mppi import AdaptiveShieldSVGMPPIParams

# Perception (optional, for /scan)
from mppi_controller.perception.obstacle_detector import ObstacleDetector
from mppi_controller.perception.obstacle_tracker import ObstacleTracker


# Error codes matching nav2_msgs/FollowPath result
NONE = 0
TF_ERROR = 100
INVALID_PATH = 101
FAILED_TO_MAKE_PROGRESS = 102
NO_VALID_CONTROL = 103


class MPPIFollowPathServer(Node):
    """Nav2-compatible FollowPath action server using MPPI.

    Replaces nav2 controller_server (C++ pluginlib) with a Python
    action server that speaks the same FollowPath action interface.
    """

    def __init__(self):
        super().__init__('mppi_follow_path_server')

        # Declare all parameters
        self._declare_parameters()

        # Get configuration
        self.odom_frame = self.get_parameter('odom_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.model_type = self.get_parameter('model_type').value
        self.controller_frequency = self.get_parameter(
            'controller_frequency').value

        # Create model and controller
        self.model = self._create_model()
        self.controller = self._create_controller()

        # Utilities
        N = self.get_parameter('N').value
        dt = self.get_parameter('dt').value
        state_dim = self.model.state_dim

        self.path_windower = PathWindower(
            horizon=N, dt=dt, state_dim=state_dim)
        self.costmap_converter = CostmapConverter(
            robot_radius=self.get_parameter('robot_radius').value,
            safety_margin=0.05,
        )
        self.goal_checker = GoalChecker(
            xy_tolerance=self.get_parameter('xy_goal_tolerance').value,
            yaw_tolerance=self.get_parameter('yaw_goal_tolerance').value,
        )
        self.progress_checker = ProgressChecker(
            required_movement=0.5,
            time_allowance=10.0,
        )

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.current_odom: Optional[Odometry] = None
        self.current_costmap: Optional[OccupancyGrid] = None

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10,
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=5,
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_reliable)

        costmap_topic = self.get_parameter('costmap_topic').value
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, costmap_topic,
            self._costmap_callback, qos_reliable)

        # Optional LaserScan
        self.obstacle_detector = None
        self.obstacle_tracker = None
        if self.get_parameter('scan_enabled').value:
            self.obstacle_detector = ObstacleDetector()
            self.obstacle_tracker = ObstacleTracker()
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self._scan_callback, qos_sensor)
            self.get_logger().info('LaserScan obstacle detection enabled')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', qos_reliable)
        self.local_plan_pub = self.create_publisher(
            Path, '/mppi/local_plan', qos_reliable)

        # Action server
        # Use nav2_msgs if available, otherwise a simple action interface
        try:
            from nav2_msgs.action import FollowPath
            self._action_type = FollowPath
        except ImportError:
            self.get_logger().warn(
                'nav2_msgs not found. Action server will not be created. '
                'Install nav2_msgs for full nav2 integration.')
            self._action_type = None

        if self._action_type is not None:
            self._action_server = ActionServer(
                self,
                self._action_type,
                'follow_path',
                execute_callback=self._execute_callback,
                goal_callback=self._goal_callback,
                cancel_callback=self._cancel_callback,
                callback_group=ReentrantCallbackGroup(),
            )

        self.get_logger().info(
            f'MPPI FollowPath Server initialized '
            f'[{self.get_parameter("controller_type").value}] '
            f'@ {self.controller_frequency:.0f}Hz')

    # ─── Parameter Declaration ────────────────────────────────────

    def _declare_parameters(self):
        """Declare all ROS2 parameters."""
        # Node
        self.declare_parameter('controller_frequency', 20.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('robot_frame', 'base_link')

        # Model
        self.declare_parameter('model_type', 'kinematic')
        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('omega_max', 1.9)

        # MPPI core
        self.declare_parameter('controller_type', 'shield')
        self.declare_parameter('N', 30)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('K', 1024)
        self.declare_parameter('lambda_', 1.0)
        self.declare_parameter('sigma', [0.5, 0.5])
        self.declare_parameter('Q', [10.0, 10.0, 1.0])
        self.declare_parameter('R', [0.1, 0.1])
        self.declare_parameter('Qf', [20.0, 20.0, 2.0])

        # CBF/Shield
        self.declare_parameter('cbf_alpha', 0.3)
        self.declare_parameter('cbf_weight', 1000.0)
        self.declare_parameter('cbf_safety_margin', 0.1)
        self.declare_parameter('cbf_use_safety_filter', False)
        self.declare_parameter('shield_enabled', True)

        # Goal/Progress
        self.declare_parameter('xy_goal_tolerance', 0.25)
        self.declare_parameter('yaw_goal_tolerance', 0.25)

        # Costmap
        self.declare_parameter('costmap_topic',
                               '/local_costmap/costmap_raw')
        self.declare_parameter('robot_radius', 0.22)

        # LaserScan
        self.declare_parameter('scan_enabled', False)

    # ─── Model & Controller Factory ───────────────────────────────

    def _create_model(self):
        """Create robot model from parameters."""
        v_max = self.get_parameter('v_max').value
        omega_max = self.get_parameter('omega_max').value

        if self.model_type == 'dynamic':
            return DifferentialDriveDynamic(
                v_max=v_max, omega_max=omega_max,
                mass=10.0, inertia=1.0, c_v=0.1, c_omega=0.1)
        else:
            return DifferentialDriveKinematic(
                v_max=v_max, omega_max=omega_max)

    def _create_controller(self):
        """Create MPPI controller from parameters (same factory as node)."""
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

        ct = self.get_parameter('controller_type').value

        if ct == 'vanilla':
            return MPPIController(self.model, MPPIParams(**params_dict))

        elif ct == 'tube':
            params = TubeMPPIParams(
                **params_dict, tube_enabled=True,
                K_fb=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
                tube_margin=0.3)
            return TubeMPPIController(self.model, params)

        elif ct == 'log':
            return LogMPPIController(
                self.model, LogMPPIParams(**params_dict, use_baseline=True))

        elif ct == 'tsallis':
            return TsallisMPPIController(
                self.model, TsallisMPPIParams(**params_dict, tsallis_q=1.2))

        elif ct == 'risk_aware':
            return RiskAwareMPPIController(
                self.model,
                RiskAwareMPPIParams(**params_dict, cvar_alpha=0.7))

        elif ct == 'smooth':
            return SmoothMPPIController(
                self.model, SmoothMPPIParams(**params_dict, jerk_weight=1.0))

        elif ct == 'svmpc':
            return SteinVariationalMPPIController(
                self.model,
                SteinVariationalMPPIParams(
                    **params_dict, svgd_num_iterations=3,
                    svgd_step_size=0.01))

        elif ct == 'spline':
            return SplineMPPIController(
                self.model,
                SplineMPPIParams(
                    **params_dict, spline_num_knots=8, spline_degree=3))

        elif ct == 'svg':
            return SVGMPPIController(
                self.model,
                SVGMPPIParams(
                    **params_dict, svg_num_guide_particles=32,
                    svgd_num_iterations=3, svg_guide_step_size=0.01))

        elif ct == 'cbf':
            cbf_kw = self._get_cbf_params()
            return CBFMPPIController(
                self.model,
                CBFMPPIParams(**params_dict, **cbf_kw))

        elif ct == 'shield':
            cbf_kw = self._get_cbf_params()
            return ShieldMPPIController(
                self.model,
                ShieldMPPIParams(
                    **params_dict, **cbf_kw,
                    shield_enabled=self.get_parameter(
                        'shield_enabled').value))

        elif ct == 'adaptive_shield':
            cbf_kw = self._get_cbf_params()
            return AdaptiveShieldMPPIController(
                self.model,
                AdaptiveShieldParams(
                    **params_dict, **cbf_kw,
                    shield_enabled=self.get_parameter(
                        'shield_enabled').value))

        elif ct == 'adaptive_shield_svg':
            cbf_kw_svg = {
                'cbf_obstacles': [],
                'cbf_safety_margin': self.get_parameter(
                    'cbf_safety_margin').value,
            }
            return AdaptiveShieldSVGMPPIController(
                self.model,
                AdaptiveShieldSVGMPPIParams(
                    **params_dict, **cbf_kw_svg,
                    svgd_num_iterations=3, svgd_step_size=0.01,
                    svg_num_guide_particles=32,
                    svg_guide_step_size=0.01,
                    shield_enabled=self.get_parameter(
                        'shield_enabled').value,
                    shield_cbf_alpha=self.get_parameter(
                        'cbf_alpha').value))

        else:
            self.get_logger().warn(
                f'Unknown controller type: {ct}, using shield')
            cbf_kw = self._get_cbf_params()
            return ShieldMPPIController(
                self.model,
                ShieldMPPIParams(
                    **params_dict, **cbf_kw, shield_enabled=True))

    def _get_cbf_params(self) -> dict:
        """Extract CBF-related parameters."""
        return {
            'cbf_obstacles': [],
            'cbf_alpha': self.get_parameter('cbf_alpha').value,
            'cbf_weight': self.get_parameter('cbf_weight').value,
            'cbf_safety_margin': self.get_parameter(
                'cbf_safety_margin').value,
            'cbf_use_safety_filter': self.get_parameter(
                'cbf_use_safety_filter').value,
        }

    # ─── Callbacks ────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        """Store latest odometry message."""
        self.current_odom = msg

    def _costmap_callback(self, msg: OccupancyGrid):
        """Store latest costmap and update obstacles."""
        self.current_costmap = msg
        self._update_obstacles_from_costmap(msg)

    def _scan_callback(self, msg: LaserScan):
        """LaserScan obstacle detection (if enabled)."""
        if self.obstacle_detector is None or self.obstacle_tracker is None:
            return

        robot_state = self._get_robot_state()
        robot_pose = robot_state[:3] if robot_state is not None else None

        detections = self.obstacle_detector.detect(
            np.array(msg.ranges),
            msg.angle_min, msg.angle_increment, robot_pose)

        dt = 1.0 / self.controller_frequency
        self.obstacle_tracker.update(detections, dt)

        if hasattr(self.controller, 'update_obstacles'):
            obstacles = self.obstacle_tracker.get_obstacles_as_tuples()
            self.controller.update_obstacles(obstacles)

    def _update_obstacles_from_costmap(self, costmap_msg: OccupancyGrid):
        """Convert costmap to circle obstacles and update controller."""
        if not hasattr(self.controller, 'update_obstacles'):
            return

        robot_state = self._get_robot_state()
        robot_x = robot_state[0] if robot_state is not None else 0.0
        robot_y = robot_state[1] if robot_state is not None else 0.0

        info = costmap_msg.info
        obstacles = self.costmap_converter.convert(
            data=np.array(costmap_msg.data, dtype=np.int16),
            width=info.width, height=info.height,
            resolution=info.resolution,
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            robot_x=robot_x, robot_y=robot_y)

        self.controller.update_obstacles(obstacles)

    # ─── Action Server Callbacks ──────────────────────────────────

    def _goal_callback(self, goal_request):
        """Accept or reject action goal."""
        self.get_logger().info(
            f'Received FollowPath goal with '
            f'{len(goal_request.path.poses)} poses')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Accept cancel requests."""
        self.get_logger().info('FollowPath cancel requested')
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        """Main control loop for FollowPath action."""
        self.get_logger().info('Executing FollowPath...')

        path_msg = goal_handle.request.path
        result = self._action_type.Result()

        # Validate path
        if len(path_msg.poses) == 0:
            self.get_logger().error('Received empty path')
            result.error_code = INVALID_PATH
            goal_handle.abort()
            return result

        # Extract global path as numpy (M, 3)
        global_path = np.array([
            pose_stamped_to_array(ps) for ps in path_msg.poses])
        goal_pose = global_path[-1]

        # Reset checkers
        self.goal_checker.reset()
        self.progress_checker.reset()
        self.path_windower.reset()

        rate_period = 1.0 / self.controller_frequency
        feedback = self._action_type.Feedback()

        while rclpy.ok():
            # Check cancel
            if goal_handle.is_cancel_requested:
                self._publish_zero_velocity()
                goal_handle.canceled()
                self.get_logger().info('FollowPath canceled')
                result.error_code = NONE
                return result

            loop_start = time.monotonic()

            # 1. Get robot state
            robot_state = self._get_robot_state()
            if robot_state is None:
                self.get_logger().warn(
                    'Failed to get robot state from TF',
                    throttle_duration_sec=2.0)
                result.error_code = TF_ERROR
                goal_handle.abort()
                return result

            current_time = self.get_clock().now().nanoseconds * 1e-9

            # 2. Check goal reached
            if self.goal_checker.is_goal_reached(robot_state, goal_pose):
                self._publish_zero_velocity()
                result.error_code = NONE
                goal_handle.succeed()
                self.get_logger().info('Goal reached!')
                return result

            # 3. Check progress (stuck detection)
            if not self.progress_checker.check_progress(
                    robot_state, current_time):
                self._publish_zero_velocity()
                self.get_logger().warn('Robot is stuck!')
                result.error_code = FAILED_TO_MAKE_PROGRESS
                goal_handle.abort()
                return result

            # 4. Transform path to odom frame
            transformed_path = self._transform_path(path_msg)
            if transformed_path is None:
                # Fallback: use global_path directly
                transformed_path = global_path

            # 5. Extract local reference via PathWindower
            reference, closest_idx = self.path_windower.extract_reference(
                transformed_path, robot_state)

            # 6. Compute control
            try:
                control, info = self.controller.compute_control(
                    robot_state, reference)
            except Exception as e:
                self.get_logger().error(
                    f'MPPI compute_control failed: {e}')
                self._publish_zero_velocity()
                result.error_code = NO_VALID_CONTROL
                goal_handle.abort()
                return result

            # 7. Publish cmd_vel
            self._publish_velocity(control)

            # 8. Publish local plan for visualization
            self._publish_local_plan(reference)

            # 9. Publish feedback
            dist_to_goal = self.goal_checker.get_distance_to_goal(
                robot_state, goal_pose)
            speed = float(np.sqrt(
                control[0] ** 2)) if len(control) > 0 else 0.0
            feedback.distance_to_goal = dist_to_goal
            feedback.speed = speed
            goal_handle.publish_feedback(feedback)

            # Rate control
            elapsed = time.monotonic() - loop_start
            sleep_time = rate_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Node shutting down
        self._publish_zero_velocity()
        result.error_code = NONE
        goal_handle.abort()
        return result

    # ─── Helper Methods ───────────────────────────────────────────

    def _get_robot_state(self) -> Optional[np.ndarray]:
        """Get robot state from TF2 + odometry."""
        try:
            tf_time = rclpy.time.Time()
            timeout = Duration(seconds=0.1)
            return get_robot_state_from_tf(
                self.tf_buffer, self.odom_frame, self.robot_frame,
                tf_time, timeout, self.model_type, self.current_odom)
        except Exception:
            return None

    def _transform_path(self, path_msg: Path) -> Optional[np.ndarray]:
        """Transform path to odom frame."""
        try:
            timeout = Duration(seconds=0.1)
            return transform_path_to_frame(
                self.tf_buffer, path_msg, self.odom_frame, timeout)
        except Exception:
            return None

    def _publish_velocity(self, control: np.ndarray):
        """Publish Twist command."""
        msg = Twist()
        msg.linear.x = float(control[0])
        msg.angular.z = float(control[1])
        self.cmd_vel_pub.publish(msg)

    def _publish_zero_velocity(self):
        """Publish zero velocity stop command."""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def _publish_local_plan(self, reference: np.ndarray):
        """Publish local plan as nav_msgs/Path for RVIZ."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.odom_frame

        for i in range(len(reference)):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(reference[i, 0])
            ps.pose.position.y = float(reference[i, 1])
            # Set orientation from theta
            theta = float(reference[i, 2]) if reference.shape[1] > 2 else 0.0
            ps.pose.orientation.z = float(np.sin(theta / 2))
            ps.pose.orientation.w = float(np.cos(theta / 2))
            path.poses.append(ps)

        self.local_plan_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = MPPIFollowPathServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
