"""
The FR3 inference stack: inference.launch.py with robot:=fr3. Kept under this name for fr3_session.sh.

    ros2 launch nuc/launch/fr3_inference.launch.py gripper:=faulhaber execute_gripper:=true execute_arm:=true

The home pose stays in ~/.ros/polyumi_home_pose.yaml here; HSA runs through inference.launch.py use their own.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

FORWARDED = {
    'robot_ip': '192.168.51.20',
    'execute_arm': 'false',
    'execute_gripper': 'false',
    'gripper': 'faulhaber',
    'max_acceleration': '1.5',
    'home_pose_file': '~/.ros/polyumi_home_pose.yaml',
}


def generate_launch_description():
    """Include inference.launch.py for the FR3, forwarding every argument it takes."""
    return LaunchDescription(
        [
            *[DeclareLaunchArgument(name, default_value=default) for name, default in FORWARDED.items()],
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(Path(__file__).resolve().parent / 'inference.launch.py')),
                launch_arguments=[('robot', 'fr3'), *((name, LaunchConfiguration(name)) for name in FORWARDED)],
            ),
        ]
    )
