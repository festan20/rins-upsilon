"""Full task2 launch (simulation).

Starts (in order):
  1. Gazebo simulator + TurtleBot4 spawn        (from dis_tutorial3)
  2. Localization (AMCL + map_server)           (factory map)
  3. Nav2                                       (from dis_tutorial3)
  4. Upsilon nodes: face_detector, ring_detector, speech, controller

Usage
-----
  ros2 launch upsilon task2.launch.py
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
    DeclareLaunchArgument(
        'blue_line_following',
        default_value='true',
        choices=['true', 'false'],
        description='Launch blue-line detector and follower',
    ),
    DeclareLaunchArgument(
        'blue_line_active',
        default_value='false',
        choices=['true', 'false'],
        description='Start blue-line follower active (normally false)',
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


def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
    pkg_dis_tutorial7 = get_package_share_directory('dis_tutorial7')
    pkg_upsilon = get_package_share_directory('upsilon')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'sim.launch.py'])
        ),
        launch_arguments=[
            ('world', LaunchConfiguration('world')),
            ('model', LaunchConfiguration('model')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
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
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
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
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='scan_to_scan_filter_chain',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_dis_tutorial3, 'config', 'laser_filter_chain.yaml']),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_upsilon, 'launch', 'localization.launch.py'])
        ),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
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
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
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
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
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
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
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
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('params_file', PathJoinSubstitution([pkg_upsilon, 'config', 'nav2.yaml'])),
        ],
    )

    face_detector = Node(
        package='upsilon',
        executable='face_detector',
        name='face_detector',
        output='screen',
        parameters=[
            {'device': ''},
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'rgb_topic': '/oakd/rgb/preview/image_raw'},
            {'depth_topic': '/oakd/rgb/preview/depth'},
            {'camera_info_topic': '/oakd/rgb/preview/camera_info'},
            {'compressed_topics': False},
        ],
    )

    ring_detector = Node(
        package='upsilon',
        executable='ring_detector2',
        name='ring_detector2',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'rgb_topic': '/oakd/rgb/preview/image_raw'},
            {'depth_topic': '/oakd/rgb/preview/depth'},
            {'camera_info_topic': '/oakd/rgb/preview/camera_info'},
            {'compressed_topics': False},
        ],
        remappings=[
            ('/ring_detector2/debug', '/ring_detector/debug'),
            ('/ring_detector2/threshold', '/ring_detector/threshold'),
            ('/ring_detector2/contour', '/ring_detector/contour'),
            ('/detected_rings2', '/detected_rings'),
            ('/ring_markers2', '/ring_markers'),
        ],
    )

    speech = Node(
        package='upsilon',
        executable='speech',
        name='speech',
        output='screen',
    )

    controller = Node(
        package='upsilon',
        executable='controller',
        name='controller',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    visualizer = Node(
        package='upsilon',
        executable='visualizer',
        name='visualizer',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'rgb_topic': '/oakd/rgb/preview/image_raw'},
            {'compressed_rgb': False},
        ],
    )

    blue_line_detector = Node(
        package='upsilon',
        executable='blue_line_detector',
        name='blue_line_detector',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'rgb_topic': '/oakd/rgb/preview/image_raw'},
            {'depth_topic': '/oakd/rgb/preview/depth'},
            {'camera_info_topic': '/oakd/rgb/preview/camera_info'},
            {'compressed_topics': False},
        ],
        condition=IfCondition(LaunchConfiguration('blue_line_following')),
    )

    blue_line_follower = Node(
        package='upsilon',
        executable='blue_line_follower',
        name='blue_line_follower',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'active': LaunchConfiguration('blue_line_active')},
            {'linear_speed': 0.24},
            {'max_angular_speed': 0.8},
        ],
        condition=IfCondition(LaunchConfiguration('blue_line_following')),
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
    ld.add_action(face_detector)
    ld.add_action(ring_detector)
    ld.add_action(speech)
    ld.add_action(controller)
    ld.add_action(visualizer)
    ld.add_action(blue_line_detector)
    ld.add_action(blue_line_follower)
    return ld
