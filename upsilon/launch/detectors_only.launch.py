"""Detectors-only launch for testing without navigation."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='upsilon',
            executable='face_detector',
            name='face_detector',
            output='screen',
            parameters=[{'device': ''}],
        ),
        Node(
            package='upsilon',
            executable='ring_detector',
            name='ring_detector',
            output='screen',
        ),
        Node(
            package='upsilon',
            executable='speech',
            name='speech',
            output='screen',
        ),
        Node(
            package='upsilon',
            executable='visualizer',
            name='visualizer',
            output='screen',
        ),
    ])
