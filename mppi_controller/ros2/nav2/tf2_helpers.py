"""TF2 helper functions for MPPI Nav2 integration.

Provides utilities for extracting robot state from TF2 transforms
and converting nav_msgs/Path to numpy arrays in a target frame.
"""

import numpy as np
from typing import Optional


def quaternion_to_yaw(q) -> float:
    """Convert quaternion to yaw angle.

    Args:
        q: Quaternion with x, y, z, w attributes.

    Returns:
        Yaw angle in radians.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def pose_stamped_to_array(pose_stamped) -> np.ndarray:
    """Convert PoseStamped to [x, y, theta] array.

    Args:
        pose_stamped: geometry_msgs/PoseStamped.

    Returns:
        (3,) array [x, y, theta].
    """
    pose = pose_stamped.pose
    x = pose.position.x
    y = pose.position.y
    theta = quaternion_to_yaw(pose.orientation)
    return np.array([x, y, theta])


def get_robot_state_from_tf(tf_buffer, odom_frame: str,
                             robot_frame: str,
                             time, timeout,
                             model_type: str = 'kinematic',
                             odom_msg=None) -> Optional[np.ndarray]:
    """Get robot state from TF2 transform.

    Args:
        tf_buffer: tf2_ros.Buffer instance.
        odom_frame: Odometry frame (e.g., 'odom').
        robot_frame: Robot base frame (e.g., 'base_link').
        time: rclpy.time.Time for lookup.
        timeout: rclpy.duration.Duration for lookup timeout.
        model_type: 'kinematic' (3D: x,y,θ) or 'dynamic' (5D: x,y,θ,v,ω).
        odom_msg: Optional Odometry message for velocity extraction.

    Returns:
        State array [x, y, θ] or [x, y, θ, v, ω], or None on failure.
    """
    try:
        transform = tf_buffer.lookup_transform(
            odom_frame, robot_frame, time, timeout)
    except Exception:
        return None

    t = transform.transform
    x = t.translation.x
    y = t.translation.y
    theta = quaternion_to_yaw(t.rotation)

    if model_type == 'kinematic':
        return np.array([x, y, theta])
    elif model_type == 'dynamic':
        v = 0.0
        omega = 0.0
        if odom_msg is not None:
            v = odom_msg.twist.twist.linear.x
            omega = odom_msg.twist.twist.angular.z
        return np.array([x, y, theta, v, omega])
    else:
        return np.array([x, y, theta])


def transform_path_to_frame(tf_buffer, path, target_frame: str,
                             timeout) -> Optional[np.ndarray]:
    """Transform nav_msgs/Path to numpy array in target frame.

    Args:
        tf_buffer: tf2_ros.Buffer instance.
        path: nav_msgs/Path message.
        target_frame: Target frame ID (e.g., 'odom').
        timeout: rclpy.duration.Duration for TF lookup.

    Returns:
        (M, 3) array of [x, y, theta] in target_frame, or None on failure.
    """
    if len(path.poses) == 0:
        return np.zeros((0, 3))

    source_frame = path.header.frame_id
    if not source_frame:
        source_frame = 'map'

    # If same frame, just extract poses
    if source_frame == target_frame:
        return _extract_poses(path)

    # Lookup transform: target_frame ← source_frame
    try:
        from rclpy.time import Time
        transform = tf_buffer.lookup_transform(
            target_frame, source_frame, Time(), timeout)
    except Exception:
        return None

    # Apply transform to all poses
    t = transform.transform
    tx = t.translation.x
    ty = t.translation.y
    yaw_offset = quaternion_to_yaw(t.rotation)
    cos_y = np.cos(yaw_offset)
    sin_y = np.sin(yaw_offset)

    poses = _extract_poses(path)
    if poses is None or len(poses) == 0:
        return poses

    # Rotate + translate
    transformed = np.zeros_like(poses)
    transformed[:, 0] = cos_y * poses[:, 0] - sin_y * poses[:, 1] + tx
    transformed[:, 1] = sin_y * poses[:, 0] + cos_y * poses[:, 1] + ty
    # Wrap angle
    transformed[:, 2] = _wrap_angle(poses[:, 2] + yaw_offset)

    return transformed


def _extract_poses(path) -> np.ndarray:
    """Extract (M, 3) [x, y, theta] array from nav_msgs/Path."""
    M = len(path.poses)
    poses = np.zeros((M, 3))
    for i, ps in enumerate(path.poses):
        poses[i] = pose_stamped_to_array(ps)
    return poses


def _wrap_angle(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return np.arctan2(np.sin(angles), np.cos(angles))
