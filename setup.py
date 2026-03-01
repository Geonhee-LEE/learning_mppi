from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'learning_mppi'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests', 'tests.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('configs/*.yaml')),
        # RVIZ config
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Geonhee Lee',
    maintainer_email='gunhee6392@gmail.com',
    description='Model Predictive Path Integral (MPPI) Control for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mppi_controller_node = mppi_controller.ros2.mppi_controller_node:main',
            'mppi_visualizer_node = mppi_controller.ros2.mppi_visualizer_node:main',
            'trajectory_publisher = mppi_controller.ros2.trajectory_publisher:main',
            'simple_robot_simulator = mppi_controller.ros2.simple_robot_simulator:main',
            'mppi_follow_path_server = mppi_controller.ros2.nav2.follow_path_server:main',
        ],
    },
)
