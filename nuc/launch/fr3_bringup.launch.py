"""
The FR3 hardware session: bringup.launch.py with robot:=fr3. Kept under this name for fr3_session.sh.

    ros2 launch nuc/launch/fr3_bringup.launch.py
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Include bringup.launch.py for the FR3, forwarding the arguments fr3_session.sh uses."""
    return LaunchDescription(
        [
            DeclareLaunchArgument('robot_ip', default_value='192.168.51.20'),
            DeclareLaunchArgument('load_gripper', default_value='false'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(Path(__file__).resolve().parent / 'bringup.launch.py')),
                launch_arguments={
                    'robot': 'fr3',
                    'robot_ip': LaunchConfiguration('robot_ip'),
                    'load_gripper': LaunchConfiguration('load_gripper'),
                }.items(),
            ),
        ]
    )
