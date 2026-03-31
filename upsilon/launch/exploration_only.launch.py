"""Exploration-only launch for testing waypoint coverage.

Starts the full simulation + Nav2 stack plus the controller node,
but WITHOUT face/ring detectors or speech.  The controller will drive
through all EXPLORATION_WAYPOINTS, spinning at each one, then stop.

Usage
-----
  ros2 launch upsilon exploration_only.launch.py
  ros2 launch upsilon exploration_only.launch.py world:=task1_green_demo
  ros2 launch upsilon exploration_only.launch.py rviz:=false
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


ARGUMENTS = [
    DeclareLaunchArgument(
        'world',
        default_value='task1_blue_demo',
        description='Gazebo world name. '
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

for axis in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(
        DeclareLaunchArgument(axis, default_value='0.0',
                              description=f'Initial robot {axis} pose')
    )


def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
    pkg_upsilon       = get_package_share_directory('upsilon')

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

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments=[
            ('namespace',    LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
        ],
    )

    controller = Node(
        package='upsilon',
        executable='controller',
        name='controller',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gazebo)
    ld.add_action(robot_spawn)
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(controller)
    return ld
