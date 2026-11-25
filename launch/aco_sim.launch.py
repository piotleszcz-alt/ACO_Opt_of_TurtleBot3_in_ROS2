"""
Uruchomienie całego projektu jedną komendą.

  ros2 launch aco_planner aco_sim.launch.py                    # arena + Gazebo
  ros2 launch aco_planner aco_sim.launch.py world:=maze        # labirynt + Gazebo
  ros2 launch aco_planner aco_sim.launch.py world:=maze gazebo:=false
                                                               # tylko planowanie w RViz2

Uruchamia: Gazebo z robotem, transformację map -> odom, aco_node, pure_pursuit, RViz2.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG_SRC = os.path.expanduser('~/ros2_ws/src/aco_planner')
TB3_GAZEBO = get_package_share_directory('turtlebot3_gazebo')

# Wszystko, co różni oba światy, w jednym miejscu.
WORLDS = {
    'arena': {
        'map_yaml': '/opt/ros/jazzy/share/turtlebot3_navigation2/map/map.yaml',
        'gazebo_launch': os.path.join(TB3_GAZEBO, 'launch', 'turtlebot3_world.launch.py'),
        'x': -2.0, 'y': -0.5,               # gdzie Gazebo stawia robota
    },
    'maze': {
        'map_yaml': os.path.join(PKG_SRC, 'maps', 'maze', 'maze.yaml'),
        'gazebo_launch': os.path.join(PKG_SRC, 'maps', 'maze', 'maze_world.launch.py'),
        'x': -1.6, 'y': -1.6,               # z maze_info.txt - zmień, jeśli wygenerujesz inny labirynt
    },
}

RVIZ_CONFIG = os.path.join(PKG_SRC, 'aco.rviz')


def launch_setup(context):
    world = LaunchConfiguration('world').perform(context)
    gazebo = LaunchConfiguration('gazebo').perform(context).lower() == 'true'
    cfg = WORLDS[world]
    sim_time = gazebo            # czas symulacji tylko wtedy, gdy Gazebo publikuje /clock

    actions = [
        Node(
            package='aco_planner', executable='aco_node', output='screen',
            parameters=[{'map_yaml': cfg['map_yaml'], 'use_sim_time': sim_time}],
        ),
    ]

    rviz_args = ['-d', RVIZ_CONFIG] if os.path.exists(RVIZ_CONFIG) else []
    actions.append(Node(
        package='rviz2', executable='rviz2', output='log',
        arguments=rviz_args,
        parameters=[{'use_sim_time': sim_time}],
    ))

    if gazebo:
        actions += [
            SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(cfg['gazebo_launch'])),
            Node(
                package='tf2_ros', executable='static_transform_publisher', output='log',
                arguments=['--x', str(cfg['x']), '--y', str(cfg['y']),
                           '--frame-id', 'map', '--child-frame-id', 'odom'],
            ),
            Node(
                package='aco_planner', executable='pure_pursuit', output='screen',
                parameters=[{'odom_offset_x': cfg['x'], 'odom_offset_y': cfg['y'],
                             'use_sim_time': sim_time}],
            ),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='arena', choices=list(WORLDS),
                              description='arena albo maze'),
        DeclareLaunchArgument('gazebo', default_value='true',
                              description='false = tylko aco_node i RViz2, bez robota'),
        OpaqueFunction(function=launch_setup),
    ])