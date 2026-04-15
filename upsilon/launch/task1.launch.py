"""Full task1 launch (real robot).

Starts (in order):
  1. Localization (AMCL + map_server, pre-built map)  (from dis_tutorial3)
  2. Nav2                                       (from dis_tutorial3)
  3. Upsilon nodes: face_detector, ring_detector, speech, controller

Usage
-----
  ros2 launch upsilon task1.launch.py
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ARGUMENTS = [
    # DeclareLaunchArgument(
    #     'world',
    #     default_value='task1',
    #     description='Gazebo world name (without .sdf). '
    #                 'Options: task1_blue_demo, task1_green_demo, task1_yellow_demo',
    # ),
    # DeclareLaunchArgument(
    #     'model',
    #     default_value='standard',
    #     choices=['standard', 'lite'],
    #     description='TurtleBot4 model',
    # ),
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
    # DeclareLaunchArgument(
    #     'use_sim_time',
    #     default_value='true',
    #     choices=['true', 'false'],
    #     description='Use simulation clock',
    # ),
]

# Initial robot pose in the world (DISABLED — real robot)
# for axis in ['x', 'y', 'z', 'yaw']:
#     ARGUMENTS.append(
#         DeclareLaunchArgument(axis, default_value='0.0',
#                               description=f'Initial robot {axis} pose')
#     )


def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
    pkg_upsilon = get_package_share_directory('upsilon')

    # ------------------------------------------------------------------
    # 1. Gazebo simulator + world (DISABLED — real robot)
    # ------------------------------------------------------------------
    # gazebo = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'sim.launch.py'])
    #     ),
    #     launch_arguments=[
    #         ('world',        LaunchConfiguration('world')),
    #         ('model',        LaunchConfiguration('model')),
    #         ('use_sim_time', 'false'),
    #     ],
    # )

    # ------------------------------------------------------------------
    # 2. Robot spawn (DISABLED — real robot)
    # ------------------------------------------------------------------
    # robot_spawn = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'turtlebot4_spawn.launch.py'])
    #     ),
    #     launch_arguments=[
    #         ('namespace',    LaunchConfiguration('namespace')),
    #         ('model',        LaunchConfiguration('model')),
    #         ('rviz',         'false'),
    #         ('use_sim_time', 'false'),
    #         ('x',            LaunchConfiguration('x')),
    #         ('y',            LaunchConfiguration('y')),
    #         ('z',            LaunchConfiguration('z')),
    #         ('yaw',          LaunchConfiguration('yaw')),
    #     ],
    # )

    # ------------------------------------------------------------------
    # 2b. RViz with custom config (includes marker array displays)
    # ------------------------------------------------------------------
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([pkg_upsilon, 'config', 'upsilon.rviz'])],
        parameters=[{'use_sim_time': False}],
    )

    # ------------------------------------------------------------------
    # 2c. Laser filter — converts /scan → /scan_filtered
    #     (AMCL subscribes to scan_filtered, see localization.yaml)
    # ------------------------------------------------------------------
    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='scan_to_scan_filter_chain',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_dis_tutorial3, 'config', 'laser_filter_chain.yaml']),
            {'use_sim_time': False},
        ],
    )

    # ------------------------------------------------------------------
    # 3. Localization — AMCL + map_server with pre-built map
    # ------------------------------------------------------------------
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_upsilon, 'launch', 'localization.launch.py'])
        ),
        launch_arguments=[
            ('namespace',    LaunchConfiguration('namespace')),
            ('use_sim_time', 'false'),
            ('map',          PathJoinSubstitution([pkg_upsilon, 'map', 'realMap.yaml'])),
            ('params',       PathJoinSubstitution([pkg_upsilon, 'config', 'localization.yaml'])),
        ],
    )

    # ------------------------------------------------------------------
    # 4. Nav2
    # ------------------------------------------------------------------
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments=[
            ('namespace',    LaunchConfiguration('namespace')),
            ('use_sim_time', 'false'),
            ('params_file',  PathJoinSubstitution([pkg_upsilon, 'config', 'nav2.yaml'])),
        ],
    )

    # ------------------------------------------------------------------
    # 5. Upsilon nodes
    # ------------------------------------------------------------------
    face_detector = Node(
        package='upsilon',
        executable='face_detector',
        name='face_detector',
        output='screen',
        parameters=[
            {'device': ''},
            {'use_sim_time': False},
        ],
        remappings=[
            ('/oakd/rgb/preview/image_raw', '/gemini/color/image_raw'),
            ('/oakd/rgb/preview/depth/points', '/gemini/depth/points'),
        ],
    )

    ring_detector = Node(
        package='upsilon',
        executable='ring_detector',
        name='ring_detector',
        output='screen',
        parameters=[{'use_sim_time': False}],
        remappings=[
            ('/oakd/rgb/preview/image_raw', '/gemini/color/image_raw'),
            ('/oakd/rgb/preview/depth/points', '/gemini/depth/points'),
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
        parameters=[{'use_sim_time': False}],
    )

    visualizer = Node(
        package='upsilon',
        executable='visualizer',
        name='visualizer',
        output='screen',
        parameters=[{'use_sim_time': False}],
        remappings=[
            ('/oakd/rgb/preview/image_raw', '/gemini/color/image_raw'),
            ('/oakd/rgb/preview/depth/points', '/gemini/depth/points'),
        ],
    )

    # ------------------------------------------------------------------
    ld = LaunchDescription(ARGUMENTS)
    # ld.add_action(gazebo)       # DISABLED — real robot
    # ld.add_action(robot_spawn)  # DISABLED — real robot
    ld.add_action(rviz2)
    ld.add_action(laser_filter)
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(face_detector)
    ld.add_action(ring_detector)
    ld.add_action(speech)
    ld.add_action(controller)
    ld.add_action(visualizer)
    return ld
