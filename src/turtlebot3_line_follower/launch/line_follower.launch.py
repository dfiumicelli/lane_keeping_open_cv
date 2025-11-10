#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot3_line_follower',
            executable='line_follower',
            name='line_follower_node',
            output='screen',
            parameters=[
                {'linear_speed': 0.2},
                {'angular_speed': 0.5},
                {'debug_mode': True}
            ]
        )
    ])
