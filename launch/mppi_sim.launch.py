#!/usr/bin/env python3
"""
MPPI Simulation Launch File

Launches:
- MPPI Controller Node
- Trajectory Publisher (reference path)
- MPPI Visualizer (RVIZ markers)
- RVIZ2 (optional)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('mppi_ros2')

    # Declare launch arguments
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    model_type = LaunchConfiguration('model_type', default='kinematic')
    controller_type = LaunchConfiguration('controller_type', default='vanilla')
    trajectory_type = LaunchConfiguration('trajectory_type', default='circle')

    # Config files
    rviz_config = os.path.join(pkg_dir, 'rviz', 'mppi_sim.rviz')
    controller_config = os.path.join(pkg_dir, 'config', 'mppi_controller.yaml')
    trajectory_config = os.path.join(pkg_dir, 'config', 'trajectory.yaml')

    # Nodes
    simple_robot_simulator_node = Node(
        package='mppi_ros2',
        executable='simple_robot_simulator',
        name='simple_robot_simulator',
        output='screen',
        parameters=[
            {
                'frame_id': 'map',
                'child_frame_id': 'base_link',
                'publish_rate': 50.0,
                'process_noise_std': 0.0,
                'v_max': 2.0,
                'omega_max': 2.0,
                'initial_x': 0.0,
                'initial_y': -5.0,
                'initial_theta': 1.5708,
            }
        ]
    )

    mppi_controller_node = Node(
        package='mppi_ros2',
        executable='mppi_controller_node',
        name='mppi_controller',
        output='screen',
        parameters=[
            controller_config,
            {
                'model_type': model_type,
                'controller_type': controller_type,
            }
        ],
        remappings=[
            ('/odom', '/odom'),
            ('/reference_path', '/reference_path'),
            ('/cmd_vel', '/cmd_vel'),
        ]
    )

    trajectory_publisher_node = Node(
        package='mppi_ros2',
        executable='trajectory_publisher',
        name='trajectory_publisher',
        output='screen',
        parameters=[
            trajectory_config,
            {
                'trajectory_type': trajectory_type,
            }
        ]
    )

    mppi_visualizer_node = Node(
        package='mppi_ros2',
        executable='mppi_visualizer_node',
        name='mppi_visualizer',
        output='screen',
        parameters=[
            {
                'frame_id': 'map',
                'visualization_rate': 10.0,
                'show_samples': True,
                'show_best': True,
                'show_reference': True,
                'max_samples_viz': 50,
            }
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
        output='screen'
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RVIZ2'
        ),
        DeclareLaunchArgument(
            'model_type',
            default_value='kinematic',
            description='Robot model type (kinematic, dynamic)'
        ),
        DeclareLaunchArgument(
            'controller_type',
            default_value='vanilla',
            description='MPPI controller type (vanilla, tube, log, tsallis, risk_aware, smooth, svmpc, spline, svg)'
        ),
        DeclareLaunchArgument(
            'trajectory_type',
            default_value='circle',
            description='Reference trajectory type (circle, figure8, sine, slalom, lemniscate, straight)'
        ),

        # Nodes
        simple_robot_simulator_node,
        mppi_controller_node,
        trajectory_publisher_node,
        mppi_visualizer_node,
        rviz_node,
    ])
