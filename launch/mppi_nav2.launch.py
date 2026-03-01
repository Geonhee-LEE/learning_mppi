#!/usr/bin/env python3
"""
MPPI Nav2 Integration Launch File

Launches the MPPI FollowPath action server for nav2 integration.
Replaces nav2's controller_server with the Python MPPI action server.

Usage:
    # With nav2 bringup (replace controller_server)
    ros2 launch learning_mppi mppi_nav2.launch.py

    # With custom config
    ros2 launch learning_mppi mppi_nav2.launch.py \
        controller_type:=adaptive_shield params_file:=/path/to/config.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('learning_mppi')

    # Launch arguments
    use_rviz = LaunchConfiguration('use_rviz', default='false')
    controller_type = LaunchConfiguration('controller_type', default='shield')
    model_type = LaunchConfiguration('model_type', default='kinematic')
    params_file = LaunchConfiguration(
        'params_file',
        default=os.path.join(pkg_dir, 'config', 'mppi_nav2.yaml'))

    # Config files
    rviz_config = os.path.join(pkg_dir, 'rviz', 'mppi_nav2.rviz')

    # MPPI FollowPath Action Server
    mppi_follow_path_node = Node(
        package='learning_mppi',
        executable='mppi_follow_path_server',
        name='mppi_follow_path_server',
        output='screen',
        parameters=[
            params_file,
            {
                'controller_type': controller_type,
                'model_type': model_type,
            },
        ],
        remappings=[
            ('/odom', '/odom'),
            ('/cmd_vel', '/cmd_vel'),
            ('/scan', '/scan'),
        ],
    )

    # RVIZ (optional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Launch RVIZ2'),
        DeclareLaunchArgument(
            'controller_type', default_value='shield',
            description='MPPI controller type'),
        DeclareLaunchArgument(
            'model_type', default_value='kinematic',
            description='Robot model type (kinematic, dynamic)'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(pkg_dir, 'config', 'mppi_nav2.yaml'),
            description='Path to MPPI nav2 config YAML'),

        mppi_follow_path_node,
        rviz_node,
    ])
