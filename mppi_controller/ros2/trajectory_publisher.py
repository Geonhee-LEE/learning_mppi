#!/usr/bin/env python3
"""
Trajectory Publisher ROS2 Node

Publishes reference trajectories for MPPI controller testing.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from typing import Callable

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


class TrajectoryPublisher(Node):
    """
    ROS2 Node for publishing reference trajectories

    Publishes:
        /reference_path (nav_msgs/Path): Reference trajectory
    """

    def __init__(self):
        super().__init__('trajectory_publisher')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self.trajectory_type = self.get_parameter('trajectory_type').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.horizon = self.get_parameter('horizon').value
        self.dt = self.get_parameter('dt').value

        # Trajectory-specific parameters
        self.radius = self.get_parameter('radius').value
        self.frequency = self.get_parameter('frequency').value
        self.amplitude = self.get_parameter('amplitude').value
        self.velocity = self.get_parameter('velocity').value

        # Time
        self.t = 0.0

        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publisher
        self.path_pub = self.create_publisher(
            Path,
            '/reference_path',
            qos_profile
        )

        # Timer
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.timer_callback
        )

        # Get trajectory generator
        self.trajectory_generator = self._get_trajectory_generator()

        self.get_logger().info(f'Trajectory Publisher initialized')
        self.get_logger().info(f'  Trajectory type: {self.trajectory_type}')
        self.get_logger().info(f'  Publish rate: {self.publish_rate} Hz')
        self.get_logger().info(f'  Horizon: {self.horizon} steps')

    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        self.declare_parameter('trajectory_type', 'circle')  # circle, figure8, sine, slalom, lemniscate, straight
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('horizon', 30)  # N
        self.declare_parameter('dt', 0.05)  # timestep

        # Trajectory-specific
        self.declare_parameter('radius', 5.0)
        self.declare_parameter('frequency', 0.1)  # Hz
        self.declare_parameter('amplitude', 3.0)
        self.declare_parameter('velocity', 1.0)  # m/s

    def _get_trajectory_generator(self) -> Callable:
        """Get trajectory generator function based on type"""
        generators = {
            'circle': self._circle_trajectory,
            'figure8': self._figure8_trajectory,
            'sine': self._sine_trajectory,
            'slalom': self._slalom_trajectory,
            'lemniscate': self._lemniscate_trajectory,
            'straight': self._straight_trajectory,
        }

        generator = generators.get(self.trajectory_type)
        if generator is None:
            self.get_logger().warn(
                f'Unknown trajectory type: {self.trajectory_type}, using circle'
            )
            return self._circle_trajectory

        return generator

    def timer_callback(self):
        """Publish reference trajectory"""
        # Generate trajectory
        poses = self.trajectory_generator(self.t, self.horizon, self.dt)

        # Create Path message
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        for pose_data in poses:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = float(pose_data[0])
            pose_stamped.pose.position.y = float(pose_data[1])
            pose_stamped.pose.position.z = 0.0

            # Convert theta to quaternion
            theta = pose_data[2]
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = float(np.sin(theta / 2.0))
            pose_stamped.pose.orientation.w = float(np.cos(theta / 2.0))

            path.poses.append(pose_stamped)

        self.path_pub.publish(path)

        # Update time
        self.t += 1.0 / self.publish_rate

    def _circle_trajectory(self, t: float, N: int, dt: float) -> np.ndarray:
        """
        Circular trajectory

        Returns:
            poses: (N+1, 3) - [x, y, theta]
        """
        poses = np.zeros((N + 1, 3))
        omega = 2 * np.pi * self.frequency  # rad/s

        for i in range(N + 1):
            t_i = t + i * dt
            angle = omega * t_i

            poses[i, 0] = self.radius * np.cos(angle)
            poses[i, 1] = self.radius * np.sin(angle)
            poses[i, 2] = angle + np.pi / 2  # Tangent direction

        return poses

    def _figure8_trajectory(self, t: float, N: int, dt: float) -> np.ndarray:
        """
        Figure-8 trajectory (Lissajous curve)

        Returns:
            poses: (N+1, 3) - [x, y, theta]
        """
        poses = np.zeros((N + 1, 3))
        omega = 2 * np.pi * self.frequency

        for i in range(N + 1):
            t_i = t + i * dt
            angle = omega * t_i

            poses[i, 0] = self.radius * np.sin(angle)
            poses[i, 1] = self.radius * np.sin(2 * angle) / 2.0

            # Compute heading from velocity
            dx = self.radius * omega * np.cos(angle)
            dy = self.radius * omega * np.cos(2 * angle)
            poses[i, 2] = np.arctan2(dy, dx)

        return poses

    def _sine_trajectory(self, t: float, N: int, dt: float) -> np.ndarray:
        """
        Sine wave trajectory

        Returns:
            poses: (N+1, 3) - [x, y, theta]
        """
        poses = np.zeros((N + 1, 3))

        for i in range(N + 1):
            t_i = t + i * dt
            x = self.velocity * t_i

            poses[i, 0] = x
            poses[i, 1] = self.amplitude * np.sin(2 * np.pi * self.frequency * x)

            # Heading
            dy_dx = self.amplitude * 2 * np.pi * self.frequency * np.cos(
                2 * np.pi * self.frequency * x
            )
            poses[i, 2] = np.arctan2(dy_dx, 1.0)

        return poses

    def _lemniscate_trajectory(self, t: float, N: int, dt: float) -> np.ndarray:
        """
        Lemniscate (∞ shape) trajectory

        Returns:
            poses: (N+1, 3) - [x, y, theta]
        """
        poses = np.zeros((N + 1, 3))
        omega = 2 * np.pi * self.frequency

        for i in range(N + 1):
            t_i = t + i * dt
            angle = omega * t_i

            # Lemniscate parametric equations
            denom = 1 + np.sin(angle) ** 2
            poses[i, 0] = self.radius * np.cos(angle) / denom
            poses[i, 1] = self.radius * np.sin(angle) * np.cos(angle) / denom

            # Compute heading
            dx = -self.radius * omega * (np.sin(angle) + np.sin(angle)**3) / denom**2
            dy = self.radius * omega * (np.cos(2*angle) - np.sin(angle)**2) / denom**2
            poses[i, 2] = np.arctan2(dy, dx)

        return poses

    def _slalom_trajectory(self, t: float, N: int, dt: float) -> np.ndarray:
        """
        Slalom trajectory (chirp sine — increasing frequency)

        Returns:
            poses: (N+1, 3) - [x, y, theta]
        """
        poses = np.zeros((N + 1, 3))
        chirp_rate = 0.008
        f0 = self.velocity / 8.0  # base_wavelength = 8.0

        for i in range(N + 1):
            t_i = t + i * dt
            x = self.velocity * t_i
            phase = 2 * np.pi * (f0 * t_i + 0.5 * chirp_rate * t_i ** 2)
            y = self.amplitude * np.sin(phase)

            # Heading via numerical differentiation
            dt_num = 0.001
            t_next = t_i + dt_num
            x_next = self.velocity * t_next
            phase_next = 2 * np.pi * (f0 * t_next + 0.5 * chirp_rate * t_next ** 2)
            y_next = self.amplitude * np.sin(phase_next)
            heading = np.arctan2(y_next - y, x_next - x)

            poses[i, 0] = x
            poses[i, 1] = y
            poses[i, 2] = heading

        return poses

    def _straight_trajectory(self, t: float, N: int, dt: float) -> np.ndarray:
        """
        Straight line trajectory

        Returns:
            poses: (N+1, 3) - [x, y, theta]
        """
        poses = np.zeros((N + 1, 3))

        for i in range(N + 1):
            t_i = t + i * dt
            poses[i, 0] = self.velocity * t_i
            poses[i, 1] = 0.0
            poses[i, 2] = 0.0  # Heading along x-axis

        return poses


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
