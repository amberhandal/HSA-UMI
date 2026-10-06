"""
Launch the arm-side inference stack: the home service, the streaming controller and, on the FR3, the gripper.

Run on the box that drives the arm, after bringup.launch.py is up:

    ros2 launch nuc/launch/inference.launch.py robot:=fer                     # loaded, nothing moves
    ros2 launch nuc/launch/inference.launch.py robot:=fer execute_arm:=true   # servo drives the arm
    ros2 launch nuc/launch/inference.launch.py robot:=fr3 execute_arm:=true gripper:=faulhaber execute_gripper:=true

The streaming Cartesian impedance controller is loaded inactive and only ACTIVATED with `execute_arm:=true`;
otherwise the arm stays on the joint-trajectory controller. Homing: the FER drives that controller directly
(fer_home_service.py); the FR3 plans through move_group (fr3_home_service.py). Separate from bringup so that
the crash-prone, FCI-gated piece restarts alone; these nodes tolerate starting before their peers, but the
arm controller must already exist or /polyumi/home has nothing to drive. See docs/hsa-arm-inference.md.
"""

from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# The home services are standalone scripts run from a plain clone (no HSA-UMI workspace on the arm box),
# so they are ExecuteProcess by path; sibling modules need nuc/ on sys.path by hand.
NUC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NUC_DIR))

import robots  # noqa: E402

GRIPPERS = ('faulhaber', 'hand', 'none')


def switch_to_servo(robot: robots.Robot) -> ExecuteProcess:
    """
    Build the controller switch that hands the arm from the trajectory controller to the servo.

    `ros2 service call`, NOT `ros2 control switch_controllers`: ros2controlcli goes through the ros2cli
    daemon, and one poisoned daemon fails this with `RuntimeError: !rclpy.ok()` while every other pane
    looks healthy. strictness 2 = STRICT: fail rather than leave two controllers claiming the same interfaces.
    activate_asap switches inside the realtime loop, where libfranka accepts the mode change (FER, Kilted).
    """
    asap = ', activate_asap: true' if robot.switch_asap else ''
    return ExecuteProcess(
        cmd=[
            'ros2',
            'service',
            'call',
            '/controller_manager/switch_controller',
            'controller_manager_msgs/srv/SwitchController',
            f"{{activate_controllers: ['{robots.SERVO_CONTROLLER}'], "
            f"deactivate_controllers: ['{robot.arm_controller}'], strictness: 2{asap}}}",
        ],
        output='screen',
    )


def gripper_drivers(robot_ip: str, gripper: str) -> list:
    """
    Start the FR3 gripper driver named by `gripper`. Both claim their hardware on startup, so at most one.

    `faulhaber` = franka_gripper_control over CANopen (needs can0 up and a completed
    /faulhaber_gripper/calibrate); `hand` = franka_hand_node, a stock Franka Hand over libfranka (bringup
    must then keep load_gripper:=false). Both serve /polyumi/target_gripper and report on
    /fr3_gripper/joint_states. See docs/crb-fr3-inference.md ("Gripper problems").
    """
    if gripper == 'hand':
        return [
            Node(
                package='franka_streaming_impedance_controller',
                executable='franka_hand_node',
                name='fr3_gripper',
                output='screen',
                # The node's own default topic is node-relative; both drivers must answer the same one.
                parameters=[{'robot_ip': robot_ip, 'target_topic': '/polyumi/target_gripper'}],
            )
        ]
    if gripper == 'faulhaber':
        return [
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare('franka_gripper_control'), 'launch', 'faulhaber_gripper.launch.xml']
                    )
                ),
                launch_arguments={'joint_state_topic': '/fr3_gripper/joint_states'}.items(),
            )
        ]
    return []


def launch_setup(context):
    """Build the stack for the robot named by the `robot` argument."""
    robot = robots.get(LaunchConfiguration('robot').perform(context))
    robot_ip = LaunchConfiguration('robot_ip').perform(context) or robot.robot_ip
    home_pose_file = LaunchConfiguration('home_pose_file').perform(context) or robot.home_pose_file
    execute_arm = LaunchConfiguration('execute_arm').perform(context) == 'true'
    execute_gripper = LaunchConfiguration('execute_gripper').perform(context) == 'true'
    gripper = LaunchConfiguration('gripper').perform(context)

    actions = [LogInfo(msg=f'[inference] robot={robot.name}; home pose file {home_pose_file}')]

    if robot.home == 'moveit':
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(NUC_DIR / 'launch' / 'fr3_move_group.launch.py')),
                launch_arguments={
                    'robot_ip': robot_ip,
                    'max_acceleration': LaunchConfiguration('max_acceleration'),
                }.items(),
            )
        )

    # Always started: it owns /polyumi/home and /polyumi/set_home, and homing borrows the arm back from
    # the servo by itself.
    actions.append(
        ExecuteProcess(
            cmd=['python3', str(NUC_DIR / robot.home_service), '--ros-args', '-p', f'home_pose_file:={home_pose_file}'],
            output='screen',
        )
    )

    # Inactive whatever execute_arm says: activating means claiming the effort interfaces the trajectory
    # controller already holds, which a spawner cannot do. That takes the switch below.
    impedance_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='polyumi_impedance_controller_spawner',
        output='screen',
        arguments=robots.spawner_arguments(
            robots.SERVO_CONTROLLER,
            robots.SERVO_TYPE,
            str(NUC_DIR / 'config' / robot.servo_params),
            inactive=True,
            timeout_s='30',
        ),
    )
    actions.append(impedance_spawner)
    # Torque control starts the moment the servo activates, so the switch exists only on execute_arm, and
    # only after the spawner has loaded the controller (switching to an unloaded one just fails).
    if execute_arm:
        actions.append(
            RegisterEventHandler(OnProcessExit(target_action=impedance_spawner, on_exit=[switch_to_servo(robot)]))
        )

    if robot.grippers and execute_gripper:
        actions += gripper_drivers(robot_ip, gripper)
    elif execute_gripper and gripper != 'none':
        actions.append(
            LogInfo(msg=f'[inference] robot={robot.name} has no gripper driver; ignoring gripper:={gripper}')
        )
    return actions


def generate_launch_description():
    """Declare the arguments; the robot-dependent actions are built at launch time by launch_setup."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'robot',
                default_value=EnvironmentVariable(robots.ROBOT_ENV, default_value=robots.DEFAULT_ROBOT),
                description=f'Which arm: {" | ".join(sorted(robots.ROBOTS))} (nuc/robots.py). '
                f'Defaults to ${robots.ROBOT_ENV}, which nuc/station_env.sh sets.',
            ),
            DeclareLaunchArgument(
                'robot_ip', default_value='', description="The arm's hostname or IP; empty = the robot's default."
            ),
            DeclareLaunchArgument(
                'home_pose_file',
                default_value='',
                description='Where /polyumi/set_home persists the taught pose and /polyumi/home reads it; empty = '
                "the robot's default (nuc/robots.py). Mock-hardware runs pass their own so they never "
                "overwrite the arm's.",
            ),
            # Defaults false: launching this file must never move the robot on its own.
            DeclareLaunchArgument(
                'execute_arm',
                default_value='false',
                description='Activate the streaming Cartesian impedance controller (MOVES THE ARM on the next chunk).',
            ),
            # Which gripper and whether it may move stay two questions: a first run pairs a live gripper with
            # a still arm. Both drivers claim their hardware on startup, so false does not start one at all.
            DeclareLaunchArgument(
                'execute_gripper',
                default_value='false',
                description='Start the driver named by `gripper` (MOVES THE FINGERS). FR3 only.',
            ),
            DeclareLaunchArgument(
                'gripper',
                default_value='faulhaber',
                choices=list(GRIPPERS),
                description='FR3 gripper driver: faulhaber (CANopen) | hand (Franka Hand) | none.',
            ),
            DeclareLaunchArgument(
                'max_acceleration',
                default_value='1.5',
                description='FR3 only: joint acceleration limit (rad/s^2) move_group times homing against; '
                'MoveIt defaults to 1 and a home sweep crawls.',
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
