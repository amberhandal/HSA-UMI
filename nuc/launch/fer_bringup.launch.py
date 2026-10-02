"""
Launch the FER hardware session: franka_bringup, the EE-frame check, the payload, then fer_arm_controller.

Run on the station, after enabling FCI in Desk (https://station.robot):

    ros2 launch nuc/launch/fer_bringup.launch.py
    ros2 launch nuc/launch/fer_bringup.launch.py desk_end_effector:=hand   # if Desk says Franka Hand

Or anywhere, without a robot (ros2_control mock hardware: no FCI, both gates skipped, no torques):

    ros2 launch nuc/launch/fer_bringup.launch.py use_fake_hardware:=true

There is no hand: the wrist bolts to fer_link8 and hsa_tcp hangs off it. fer_hand_tcp, the frame the
streaming controller takes O_T_EE in, is still republished statically to match the end effector Desk has
configured, and verified against the robot before the payload goes in (nuc/hsa_calib.py). The home service
and the impedance spawner live in fer_inference.launch.py so this crash-prone, FCI-gated piece restarts
alone. See docs/hsa-arm-inference.md for the full bringup order and its gotchas.
"""

from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node

# ros2 launch loads this file by path without touching sys.path, so a sibling import needs nuc/ added by hand.
NUC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NUC_DIR))

import hsa_calib  # noqa: E402

CALIB_MODULE = 'hsa_calib'
TRAJECTORY_CONTROLLER = 'fer_arm_controller'

# How long the spawner waits for controller_manager: the default 10 s is tight on a cold robot.
CONTROLLER_MANAGER_TIMEOUT_S = '60'
# Budget for check_ee_frame.py and set_payload.py, each of which waits on the same thing coming up.
CHECK_TIMEOUT_S = CONTROLLER_MANAGER_TIMEOUT_S


def gate(next_actions, message):
    """
    Build an OnProcessExit handler that starts `next_actions` on exit 0 and aborts the launch otherwise.

    Abort rather than warn: each gate guards something that cannot be fixed once the next step has run,
    and whose symptom on the arm is quiet. Raised (not a Shutdown event) so `ros2 launch` exits non-zero.

    :param next_actions: launch actions to return when the gated process exited 0.
    :param message: the RuntimeError text; ``{code}`` is replaced with the exit status.
    """

    def _on_exit(event, context):
        if event.returncode == 0:
            return list(next_actions)
        # The launch is already going down: the gated process was interrupted, not refused, and raising
        # would print a false diagnosis on top of the real one. 130 = KeyboardInterrupt, negative = signal.
        if event.returncode == 130 or event.returncode < 0:
            return []
        raise RuntimeError(message.format(code=event.returncode))

    return _on_exit


def controller_manager_yaml(use_fake_hardware) -> PythonExpression:
    """
    Return the fer_arm_controller parameter file name for the hardware in use, as a launch substitution.

    fer_arm_controller.yaml commands effort, which the robot needs and which lets the streaming controller
    take over the same interfaces; the .mock variant commands position, the only interface mock hardware tracks.
    """
    return PythonExpression(
        ["'fer_arm_controller.mock.yaml' if '", use_fake_hardware, "' == 'true' else 'fer_arm_controller.yaml'"]
    )


def generate_launch_description():
    """Include franka_bringup, verify the EE frame, push the payload, then spawn fer_arm_controller."""
    robot_ip = LaunchConfiguration('robot_ip')
    arm_id = LaunchConfiguration('arm_id')
    load_gripper = LaunchConfiguration('load_gripper')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    desk_end_effector = LaunchConfiguration('desk_end_effector')

    on_robot = UnlessCondition(use_fake_hardware)
    on_mock = IfCondition(use_fake_hardware)

    # The fer_hand_tcp we publish is only right if it matches the F_T_EE the robot applies; a mismatch
    # exits 1 and aborts bringup below. Ahead of set_payload: past a wrong frame the controller would
    # servo about a point 10 cm from where the policy thinks the plate is, and nothing would say so.
    check_ee_frame = ExecuteProcess(
        cmd=['python3', str(NUC_DIR / 'check_ee_frame.py'), CHECK_TIMEOUT_S, '--calib', CALIB_MODULE,
             '--desk-end-effector', desk_end_effector],
        output='screen',
        condition=on_robot,
    )  # fmt: skip

    # Tell the FCI what is bolted to the flange so its gravity compensation cancels the whole weight.
    # This is the ONLY window: the robot refuses SetLoad once a controller holds the arm, hence the
    # ordering ahead of the spawner (nuc/hsa_calib.py). Started by the EE-frame gate, never directly.
    set_load = ExecuteProcess(
        cmd=['python3', str(NUC_DIR / 'set_payload.py'), CHECK_TIMEOUT_S, '--calib', CALIB_MODULE],
        output='screen',
    )

    # The joint-trajectory controller the home service moves the arm through; franka.launch.py spawns
    # only the two broadcasters. The parameter file carries the controller's `type` too: this spawner
    # has no -t flag (see nuc/config/fer_arm_controller.yaml).
    def arm_controller_spawner(condition=None) -> Node:
        return Node(
            package='controller_manager',
            executable='spawner',
            name='fer_arm_controller_spawner',
            output='screen',
            arguments=[
                TRAJECTORY_CONTROLLER,
                '--param-file',
                PathJoinSubstitution([str(NUC_DIR / 'config'), controller_manager_yaml(use_fake_hardware)]),
                # Activate inside the realtime loop, between read() and write(). Requested from the
                # non-realtime thread (the default), the switch to torque control collides with the read
                # in flight and libfranka refuses it; the controller then shows active anyway.
                '--switch-asap',
                '--controller-manager-timeout',
                CONTROLLER_MANAGER_TIMEOUT_S,
            ],
            condition=condition,
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'robot_ip',
                default_value='panda0.robot',
                description="Hostname or IP of the FER (the station's dnsmasq name; 192.168.18.10).",
            ),
            DeclareLaunchArgument(
                'arm_id', default_value='fer', description='Arm type; drives every fer_* frame and topic name.'
            ),
            # false: there is no Franka Hand on this arm, and the flag also reaches `xacro hand:=`, so
            # the URDF carries no hand links (fer_hand_tcp is republished anyway; see the docstring).
            DeclareLaunchArgument(
                'load_gripper',
                default_value='false',
                description='Start franka_gripper and put a hand in the URDF. This arm has no hand.',
            ),
            DeclareLaunchArgument(
                'use_fake_hardware',
                default_value='false',
                description='ros2_control mock hardware instead of the robot: no FCI, no gates, no torques. '
                'The same session on any machine with franka_ros2 built.',
            ),
            DeclareLaunchArgument(
                'desk_end_effector',
                default_value=hsa_calib.DESK_END_EFFECTOR,
                choices=sorted(hsa_calib.DESK_EE_TRANSFORMS),
                description='Which end effector Desk is configured with; sets fer_link8 -> fer_hand_tcp '
                '(the frame franka reports O_T_EE in) and is verified against the robot at bringup.',
            ),
            # Our copy of franka_bringup's franka.launch.py: same nodes, same arguments, our
            # controller-manager parameters (see the note at the top of fer_hardware.launch.py).
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(NUC_DIR / 'launch' / 'fer_hardware.launch.py')),
                launch_arguments={
                    'robot_ip': robot_ip,
                    'arm_id': arm_id,
                    'load_gripper': load_gripper,
                    'use_fake_hardware': use_fake_hardware,
                }.items(),
            ),
            # On the robot: check_ee_frame -> set_load -> spawner, each step started by the one
            # before it and each failure aborting the launch. Both orderings are load-bearing.
            check_ee_frame,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=check_ee_frame,
                    on_exit=gate(
                        [set_load],
                        '\n!!! EE FRAME CHECK FAILED (exit {code}): the fer_hand_tcp this launch publishes does not\n'
                        '!!! match the end effector configured in Desk. The streaming controller would servo about\n'
                        '!!! the wrong point. Relaunch with desk_end_effector:=none|hand to match Desk, or change it\n'
                        '!!! in Desk. See nuc/hsa_calib.py ("Where fer_hand_tcp comes from").\n',
                    ),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=set_load,
                    on_exit=gate(
                        [arm_controller_spawner()],
                        '\n!!! SetLoad FAILED (exit {code}) -- the payload did not configure.\n'
                        f'!!! NOT spawning {TRAJECTORY_CONTROLLER}: the impedance controller would fight gravity it\n'
                        '!!! does not know about, and setLoad cannot be retried once a controller holds the arm.\n'
                        '!!! The real reason is in the /service_server log, not the response above:\n'
                        '!!!   grep -i "command exception" $(ls -t ~/.ros/log/ros2_control_node_*.log | head -1)\n'
                        '!!! See docs/calibration-instructions.md ("End-effector payload") and nuc/hsa_calib.py.\n',
                    ),
                )
            ),
            # On mock hardware: nothing to check and nothing to push, so the spawner runs directly.
            arm_controller_spawner(condition=on_mock),
            # The frame the policy speaks in, published whenever the arm is -- Foxglove and
            # `tf2_echo fer_link0 hsa_tcp` check the calibration without the inference launch.
            LogInfo(msg=f'[fer_bringup] TCP {hsa_calib.describe()}'),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='hsa_tcp_static_tf',
                output='screen',
                arguments=hsa_calib.static_transform_publisher_args(),
            ),
            # franka's own EE frame, which the URDF lost with the hand: one publisher per Desk setting,
            # exactly one enabled. Two would leave TF picking arbitrarily; none would stop the servo activating.
            *[
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name=f'fer_hand_tcp_static_tf_{setting}',
                    output='screen',
                    arguments=hsa_calib.ee_transform_publisher_args(setting),
                    condition=IfCondition(PythonExpression(["'", desk_end_effector, f"' == '{setting}'"])),
                )
                for setting in sorted(hsa_calib.DESK_EE_TRANSFORMS)
            ],
        ]
    )
