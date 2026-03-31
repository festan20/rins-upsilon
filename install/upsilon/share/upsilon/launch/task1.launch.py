"""Full task1 launch.

Starts (in order):
  1. Gazebo with the competition world
  2. TurtleBot4 spawn + RViz + ros_gz bridges  (from dis_tutorial3)
  3. Localization (AMCL + map_server, pre-built map)  (from dis_tutorial3)
  4. Nav2                                       (from dis_tutorial3)
  5. Upsilon nodes: face_detector, ring_detector, speech, controller

Usage
-----
  ros2 launch upsilon task1.launch.py
  ros2 launch upsilon task1.launch.py world:=task1_green_demo
  ros2 launch upsilon task1.launch.py world:=task1_yellow_demo rviz:=false
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ARGUMENTS = [
    DeclareLaunchArgument(
        'world',
        default_value='task1_blue_demo',
        description='Gazebo world name (without .sdf). '
                    'Options: task1_blue_demo, task1_green_demo, task1_yellow_demo',
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
]

# Initial robot pose in the world
for axis in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(
        DeclareLaunchArgument(axis, default_value='0.0',
                              description=f'Initial robot {axis} pose')
    )


def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
    pkg_upsilon = get_package_share_directory('upsilon')

    # ------------------------------------------------------------------
    # 1. Gazebo simulator + world
    # ------------------------------------------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'sim.launch.py'])
        ),
        launch_arguments=[
            ('world',        LaunchConfiguration('world')),
            ('model',        LaunchConfiguration('model')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
        ],
    )

    # ------------------------------------------------------------------
    # 2. Robot spawn: URDF, ros_gz bridge, TurtleBot4 nodes, RViz, TF
    # ------------------------------------------------------------------
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'turtlebot4_spawn.launch.py'])
        ),
        launch_arguments=[
            ('namespace',    LaunchConfiguration('namespace')),
            ('model',        LaunchConfiguration('model')),
            ('rviz',         LaunchConfiguration('rviz')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('x',            LaunchConfiguration('x')),
            ('y',            LaunchConfiguration('y')),
            ('z',            LaunchConfiguration('z')),
            ('yaw',          LaunchConfiguration('yaw')),
        ],
    )

    # ------------------------------------------------------------------
    # 3. Localization — AMCL + map_server with pre-built map
    # ------------------------------------------------------------------
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'localization.launch.py'])
        ),
        launch_arguments=[
            ('namespace',    LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('map',          PathJoinSubstitution([pkg_upsilon, 'map', 'map.yaml'])),
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
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
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
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )

    ring_detector = Node(
        package='upsilon',
        executable='ring_detector',
        name='ring_detector',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
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

    # ------------------------------------------------------------------
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gazebo)
    ld.add_action(robot_spawn)
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(face_detector)
    ld.add_action(ring_detector)
    ld.add_action(speech)
    ld.add_action(controller)
    return ld
