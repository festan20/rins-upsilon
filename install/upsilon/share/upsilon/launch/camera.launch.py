"""Camera + ring + face detector launch.

Starts:
  - camera_viewer  (OpenCV windows: Camera POV, Ring Detection, Threshold,
                   Contours, Face Detection)
  - ring_detector2 (topology-based ring detection)
  - face_detector  (YOLOv8 person detection on the poster faces)

Usage
-----
  ros2 launch upsilon camera.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    camera_viewer = Node(
        package='upsilon',
        executable='camera_viewer',
        name='camera_viewer',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    ring_detector = Node(
        package='upsilon',
        executable='ring_detector2',
        name='ring_detector2',
        output='screen',
        parameters=[{'use_sim_time': False}],
        # Remap v2 topics so the existing camera_viewer subscriptions pick them up
        remappings=[
            ('/ring_detector2/debug',     '/ring_detector/debug'),
            ('/ring_detector2/threshold', '/ring_detector/threshold'),
            ('/ring_detector2/contour',   '/ring_detector/contour'),
            ('/detected_rings2',          '/detected_rings'),
            ('/ring_markers2',            '/ring_markers'),
        ],
    )

    face_detector = Node(
        package='upsilon',
        executable='face_detector',
        name='face_detector',
        output='screen',
        parameters=[
            {'device': ''},
            {'use_sim_time': False},
        ],
    )

    ld = LaunchDescription()
    ld.add_action(camera_viewer)
    ld.add_action(ring_detector)
    ld.add_action(face_detector)
    return ld
