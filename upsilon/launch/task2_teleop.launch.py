"""Task2 manual-navigation launch (simulation) — Task 2 detectors.

Brings up the world + map + full Nav2 (with costmaps + keepout filter) so you can
drive the robot by clicking the "Nav2 Goal" tool in RViz. The robot stays still
until YOU send a goal. No autonomous controller, no blue-line nodes, no speech.

Perception: the three from-scratch Task 2 detectors —
  face_detector_task2, ring_detector_task2, cylinder_detector_task2 —
each publishing markers to the map (/face_markers_task2, /ring_markers_task2,
/cylinder_markers_task2) plus annotated debug images shown by the visualizer
(camera POV + per-detector windows).

Starts (in order):
  1. Gazebo simulator + TurtleBot4 spawn        (from dis_tutorial3)
  2. Localization (AMCL + map_server)           (factory map)
  3. Keepout costmap filter                      (mask + filter info)
  4. Nav2                                        (from dis_tutorial3)
  5. Task 2 detectors + visualizer

Usage
-----
  ros2 launch upsilon task2_teleop.launch.py
  ros2 launch upsilon task2_teleop.launch.py world:=task2_blue_demo

MOVING top camera:
ros2 topic pub --once /arm_command std_msgs/msg/String "data: 'manual:[0.0, 0.6, 0.5, 2.0]'"

"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


ARGUMENTS = [
    DeclareLaunchArgument(
        'world',
        default_value='task2',
        description='Gazebo world name (without .sdf). '
                    'Options: task2_blue_demo, task2_green_demo, task2_yellow_demo',
    ),
    DeclareLaunchArgument(
        'model',
        default_value='standard',
        choices=['standard', 'lite'],
        description='TurtleBot4 model',
    ),
    DeclareLaunchArgument(
        'rviz',
        default_value='true',
        choices=['true', 'false'],
        description='Launch RViz',
    ),
    DeclareLaunchArgument(
        'visualizer',
        default_value='true',
        choices=['true', 'false'],
        description='Launch the OpenCV camera-POV / detector debug windows',
    ),
    DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace',
    ),
    DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        choices=['true', 'false'],
        description='Use simulation clock',
    ),
]

SPAWN_DEFAULTS = {
    'x': '0.03770426660776138',
    'y': '0.27005261182785034',
    'z': '0.002471923828125',
    'yaw': '3.141592653589793',
}

for axis in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(
        DeclareLaunchArgument(
            axis,
            default_value=SPAWN_DEFAULTS[axis],
            description=f'Initial robot {axis} pose',
        )
    )

# Shared camera parameters for all three detectors.
CAM_PARAMS = [
    {'rgb_topic': '/oakd/rgb/preview/image_raw'},
    {'depth_topic': '/oakd/rgb/preview/depth'},
    {'camera_info_topic': '/oakd/rgb/preview/camera_info'},
    {'compressed_topics': False},
]


def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
    pkg_dis_tutorial7 = get_package_share_directory('dis_tutorial7')
    pkg_upsilon = get_package_share_directory('upsilon')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # ---------------- simulation + robot ----------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'sim.launch.py'])
        ),
        launch_arguments=[
            ('world', LaunchConfiguration('world')),
            ('model', LaunchConfiguration('model')),
            ('use_sim_time', use_sim_time),
        ],
    )

    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial7, 'launch', 'turtlebot4_spawn.launch.py'])
        ),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('model', LaunchConfiguration('model')),
            ('rviz', 'false'),
            ('use_sim_time', use_sim_time),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw')),
        ],
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([pkg_upsilon, 'config', 'upsilon.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='scan_to_scan_filter_chain',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_dis_tutorial3, 'config', 'laser_filter_chain.yaml']),
            {'use_sim_time': use_sim_time},
        ],
    )

    # ---------------- localization + Nav2 ----------------
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_upsilon, 'launch', 'localization.launch.py'])
        ),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', use_sim_time),
            ('map', PathJoinSubstitution([pkg_dis_tutorial3, 'maps', 'factory.yaml'])),
            ('params', PathJoinSubstitution([pkg_upsilon, 'config', 'localization.yaml'])),
        ],
    )

    keepout_mask_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='keepout_mask_server',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'yaml_filename': PathJoinSubstitution([pkg_upsilon, 'config', 'keepout_mask.yaml'])},
            {'topic_name': '/keepout_filter_mask'},
            {'frame_id': 'map'},
        ],
    )

    keepout_filter_info_server = Node(
        package='nav2_map_server',
        executable='costmap_filter_info_server',
        name='costmap_filter_info_server',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'type': 0},
            {'filter_info_topic': '/costmap_filter_info'},
            {'mask_topic': '/keepout_filter_mask'},
            {'base': 0.0},
            {'multiplier': 1.0},
        ],
    )

    keepout_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_keepout',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['keepout_mask_server', 'costmap_filter_info_server']},
        ],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', use_sim_time),
            ('params_file', PathJoinSubstitution([pkg_upsilon, 'config', 'nav2.yaml'])),
        ],
    )

    # ---------------- arm control ----------------
    arm_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial7, 'launch', 'control.launch.py'])
        ),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
        ],
    )

    arm_mover = Node(
        package='dis_tutorial7',
        executable='arm_mover_actions.py',
        name='arm_mover_actions',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    tile_detection = Node(
        package='upsilon',
        executable='tile_detection',
        name='tile_detection',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'rgb_topic': '/top_camera/rgb/preview/image_raw'},
        ],
    )

    anomaly_detector = Node(
        package='upsilon',
        executable='anomaly_detector',
        name='anomaly_detector',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'checkpoint': '/home/upsilon/colcon_ws/rins-upsilon/upsilon/checkpoints/anomaly_best.pth'},
            {'encoder': 'efficientnet-b0'},
            {'threshold': 0.5},
        ],
    )

    # ---------------- Task 2 detectors ----------------
    face_detector = Node(
        package='upsilon',
        executable='face_detector_task2',
        name='face_detector_task2',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}, *CAM_PARAMS],
    )

    ring_detector = Node(
        package='upsilon',
        executable='ring_detector_task2',
        name='ring_detector_task2',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}, *CAM_PARAMS],
    )

    cylinder_detector = Node(
        package='upsilon',
        executable='cylinder_detector_task2',
        name='cylinder_detector_task2',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}, *CAM_PARAMS],
    )

    visualizer = Node(
        package='upsilon',
        executable='visualizer',
        name='visualizer',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'rgb_topic': '/oakd/rgb/preview/image_raw'},
            {'compressed_rgb': False},
            {'top_rgb_topic': '/top_camera/rgb/preview/image_raw'},
            {'top_compressed_rgb': False},
            {'face_debug_topic': '/face_detector_task2/debug'},
            {'ring_debug_topic': '/ring_detector_task2/debug'},
            {'cylinder_debug_topic': '/cylinder_detector_task2/debug'},
            {'anomaly_debug_topic': '/anomaly_detection/debug'},
        ],
        condition=IfCondition(LaunchConfiguration('visualizer')),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gazebo)
    ld.add_action(rviz2)
    ld.add_action(robot_spawn)
    ld.add_action(laser_filter)
    ld.add_action(localization)
    ld.add_action(keepout_mask_server)
    ld.add_action(keepout_filter_info_server)
    ld.add_action(keepout_lifecycle_manager)
    ld.add_action(nav2)
    ld.add_action(arm_control)
    ld.add_action(arm_mover)
    ld.add_action(tile_detection)
    ld.add_action(anomaly_detector)
    ld.add_action(face_detector)
    ld.add_action(ring_detector)
    ld.add_action(cylinder_detector)
    ld.add_action(visualizer)
    return ld
