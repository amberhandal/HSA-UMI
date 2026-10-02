#  Copyright (c) 2024 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# HSA-UMI note: franka_bringup/launch/franka.launch.py from NU-MSR's franka_ros2 fork (branch panda)
# with three changes, so the two files can be diffed when the fork moves:
#   1. The controller manager takes OUR parameter file (`controllers_yaml`, default
#      nuc/config/controller_manager.yaml), which adds `overruns.manage: false`. From Jazzy on the
#      manager re-times its loop after every late cycle and drifts against franka_hardware's blocking
#      1 kHz read() until torque commands miss the robot's window (ros2_control issue #2529; NU-MSR's
#      real.launch.py carries the same setting). franka.launch.py offers no way to pass it in.
#   2. arm_id defaults to fer and load_gripper to false.
#   3. No RViz node: it runs on the laptop (hsa_ros2 rviz.launch.xml), never on the station.

"""Launch the FER hardware: robot_state_publisher, ros2_control_node and the two broadcasters."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import xacro

NUC_DIR = Path(__file__).resolve().parent.parent


def robot_description_dependent_nodes_spawner(
    context: LaunchContext, robot_ip, arm_id, use_fake_hardware, fake_sensor_commands, load_gripper
):
    """Render the URDF for these arguments and return robot_state_publisher + ros2_control_node."""
    robot_ip_str = context.perform_substitution(robot_ip)
    arm_id_str = context.perform_substitution(arm_id)
    use_fake_hardware_str = context.perform_substitution(use_fake_hardware)
    fake_sensor_commands_str = context.perform_substitution(fake_sensor_commands)
    load_gripper_str = context.perform_substitution(load_gripper)

    franka_xacro_filepath = os.path.join(
        get_package_share_directory('franka_description'), 'robots', arm_id_str, arm_id_str + '.urdf.xacro'
    )
    robot_description = xacro.process_file(
        franka_xacro_filepath,
        mappings={
            'ros2_control': 'true',
            'arm_id': arm_id_str,
            'robot_ip': robot_ip_str,
            'hand': load_gripper_str,
            'use_fake_hardware': use_fake_hardware_str,
            'fake_sensor_commands': fake_sensor_commands_str,
        },
    ).toprettyxml(indent='  ')

    franka_controllers = LaunchConfiguration('controllers_yaml')

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[franka_controllers, {'arm_id': arm_id}, {'robot_description': robot_description}],
            remappings=[('joint_states', 'franka/joint_states')],
            output={
                'stdout': 'screen',
                'stderr': 'screen',
            },
            on_exit=Shutdown(),
        ),
    ]


def generate_launch_description():
    """Declare the arguments and the nodes of the hardware session."""
    arm_id_parameter_name = 'arm_id'
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'

    arm_id = LaunchConfiguration(arm_id_parameter_name)
    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)

    robot_description_dependent_nodes_spawner_opaque_function = OpaqueFunction(
        function=robot_description_dependent_nodes_spawner,
        args=[robot_ip, arm_id, use_fake_hardware, fake_sensor_commands, load_gripper],
    )

    launch_description = LaunchDescription(
        [
            DeclareLaunchArgument(robot_ip_parameter_name, description='Hostname or IP address of the robot.'),
            DeclareLaunchArgument(
                arm_id_parameter_name,
                default_value='fer',
                description='ID of the type of arm used. Supported values: fer, fr3, fp3',
            ),
            DeclareLaunchArgument(
                'controllers_yaml',
                default_value=str(NUC_DIR / 'config' / 'controller_manager.yaml'),
                description='Parameter file for ros2_control_node (controller types + overrun handling).',
            ),
            DeclareLaunchArgument(
                use_fake_hardware_parameter_name, default_value='false', description='Use fake hardware'
            ),
            DeclareLaunchArgument(
                fake_sensor_commands_parameter_name,
                default_value='false',
                description='Fake sensor commands. Only valid when "{}" is true'.format(
                    use_fake_hardware_parameter_name
                ),
            ),
            DeclareLaunchArgument(
                load_gripper_parameter_name,
                default_value='false',
                description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                'without an end-effector.',
            ),
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher',
                parameters=[{'source_list': ['franka/joint_states', 'franka_gripper/joint_states'], 'rate': 30}],
            ),
            robot_description_dependent_nodes_spawner_opaque_function,
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster'],
                output='screen',
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['franka_robot_state_broadcaster'],
                output='screen',
                condition=UnlessCondition(use_fake_hardware),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [PathJoinSubstitution([FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])]
                ),
                launch_arguments={
                    robot_ip_parameter_name: robot_ip,
                    use_fake_hardware_parameter_name: use_fake_hardware,
                }.items(),
                condition=IfCondition(load_gripper),
            ),
        ]
    )

    return launch_description
