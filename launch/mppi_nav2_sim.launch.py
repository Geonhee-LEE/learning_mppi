#!/usr/bin/env python3
"""
MPPI Nav2 Simulation Launch File

Launches a complete simulation environment with:
- Simple robot simulator (DiffDrive)
- MPPI FollowPath action server
- MPPI Visualizer
- RVIZ2 (optional)

Usage:
    ros2 launch learning_mppi mppi_nav2_sim.launch.py
    ros2 launch learning_mppi mppi_nav2_sim.launch.py controller_type:=adaptive_shield
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
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    controller_type = LaunchConfiguration('controller_type', default='shield')
    model_type = LaunchConfiguration('model_type', default='kinematic')

    # Config files
    nav2_config = os.path.join(pkg_dir, 'config', 'mppi_nav2.yaml')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'mppi_nav2.rviz')

    # Simple Robot Simulator
    simulator_node = Node(
        package='learning_mppi',
        executable='simple_robot_simulator',
        name='simple_robot_simulator',
        output='screen',
        parameters=[{
            'frame_id': 'odom',
            'child_frame_id': 'base_link',
            'publish_rate': 50.0,
            'process_noise_std': 0.0,
            'v_max': 2.0,
            'omega_max': 2.0,
            'initial_x': 0.0,
            'initial_y': 0.0,
            'initial_theta': 0.0,
        }],
    )

    # MPPI FollowPath Action Server
    mppi_follow_path_node = Node(
        package='learning_mppi',
        executable='mppi_follow_path_server',
        name='mppi_follow_path_server',
        output='screen',
        parameters=[
            nav2_config,
            {
                'controller_type': controller_type,
                'model_type': model_type,
                'odom_frame': 'odom',
                'robot_frame': 'base_link',
            },
        ],
    )

    # MPPI Visualizer
    visualizer_node = Node(
        package='learning_mppi',
        executable='mppi_visualizer_node',
        name='mppi_visualizer',
        output='screen',
        parameters=[{
            'frame_id': 'odom',
            'visualization_rate': 10.0,
            'show_samples': True,
            'show_best': True,
            'show_reference': True,
            'max_samples_viz': 50,
        }],
    )

    # RVIZ
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
            'use_rviz', default_value='true',
            description='Launch RVIZ2'),
        DeclareLaunchArgument(
            'controller_type', default_value='shield',
            description='MPPI controller type'),
        DeclareLaunchArgument(
            'model_type', default_value='kinematic',
            description='Robot model type'),

        simulator_node,
        mppi_follow_path_node,
        visualizer_node,
        rviz_node,
    ])
