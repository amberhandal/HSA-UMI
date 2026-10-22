"""
Launch the arm's hardware session: franka bringup, the bringup gates, the payload, then the arm controller.

Run on the box that drives the arm, after enabling FCI in Desk:

    ros2 launch nuc/launch/bringup.launch.py robot:=fer      # the station's FER (default: $HSA_ROBOT, else fer)
    ros2 launch nuc/launch/bringup.launch.py robot:=fr3      # the CRB NUC's FR3
    ros2 launch nuc/launch/bringup.launch.py robot:=fer desk_end_effector:=hand   # if Desk says Franka Hand
    ros2 launch nuc/launch/bringup.launch.py robot:=fer use_fake_hardware:=true   # no robot, anywhere

What differs between the arms (IP, frames, calibration module, controller files) comes from nuc/robots.py.
On the robot the arm controller only comes up through the gates: [check_ee_frame ->] set_payload -> spawner,
each failure aborting the launch. The home service and the streaming controller live in inference.launch.py,
so this crash-prone, FCI-gated piece restarts alone. See docs/hsa-arm-inference.md.
"""

import importlib
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
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# ros2 launch loads this file by path without touching sys.path, so a sibling import needs nuc/ added by hand.
NUC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NUC_DIR))

import hsa_calib  # noqa: E402
import robots  # noqa: E402

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


def param_file(name: str):
    """Resolve a robots.py parameter-file entry: a file in nuc/config, or ``package://<pkg>/<path>``."""
    if name.startswith('package://'):
        package, _, rel = name[len('package://') :].partition('/')
        return PathJoinSubstitution([FindPackageShare(package), rel])
    return str(NUC_DIR / 'config' / name)


def hardware(robot: robots.Robot, robot_ip: str, load_gripper: str, use_fake_hardware: str):
    """
    Include franka's hardware launch (robot_state_publisher, ros2_control_node, the broadcasters).

    The FER takes our copy, nuc/launch/fer_hardware.launch.py, for its overrun setting (see the note at
    its top); the FR3 takes the franka_bringup installed on its NUC.
    """
    if robot.hardware_launch == 'nuc':
        source = PythonLaunchDescriptionSource(str(NUC_DIR / 'launch' / 'fer_hardware.launch.py'))
    else:
        source = PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])]
        )
    arguments = {'robot_ip': robot_ip, 'arm_id': robot.arm_id, 'load_gripper': load_gripper}
    if robot.hardware_launch == 'nuc':
        # franka_robot_state_broadcaster only comes up on the robot, so mock hardware must say so.
        arguments['use_fake_hardware'] = use_fake_hardware
    return IncludeLaunchDescription(source, launch_arguments=arguments.items())


def static_tf(name: str, args: list[str]) -> Node:
    """Build a tf2_ros static_transform_publisher."""
    return Node(package='tf2_ros', executable='static_transform_publisher', name=name, output='screen', arguments=args)


def flange_frames(robot: robots.Robot, desk_end_effector: str, load_gripper: bool) -> list[Node]:
    """
    Republish the fixed frames the URDF lost with load_gripper:=false.

    The streaming controller takes O_T_EE as the pose of <arm_id>_hand_tcp, so that frame must exist.
    FER: no hand, so <arm_id>_link8 -> <arm_id>_hand_tcp is published to match the Desk end-effector
    setting (verified against the robot by check_ee_frame.py). FR3: the hand is there but franka_gripper
    is not, so the whole hand subtree is republished (nuc/tcp_calib.py).
    """
    if robot.flange_frames == 'desk_ee':
        return [
            static_tf(
                f'{robot.arm_id}_hand_tcp_static_tf_{desk_end_effector}',
                hsa_calib.ee_transform_publisher_args(desk_end_effector),
            )
        ]
    if load_gripper:
        return []
    calib = importlib.import_module(robot.calib)
    return [static_tf(name, args) for name, args in calib.hand_transform_publishers()]


def launch_setup(context):
    """Build the session for the robot named by the `robot` argument."""
    robot = robots.get(LaunchConfiguration('robot').perform(context))
    calib = importlib.import_module(robot.calib)
    robot_ip = LaunchConfiguration('robot_ip').perform(context) or robot.robot_ip
    load_gripper = LaunchConfiguration('load_gripper').perform(context)
    use_fake_hardware = LaunchConfiguration('use_fake_hardware').perform(context)
    desk_end_effector = LaunchConfiguration('desk_end_effector').perform(context)
    on_mock = use_fake_hardware == 'true'

    if on_mock and robot.arm_controller_mock_params is None:
        raise RuntimeError(f'robot:={robot.name} has no mock-hardware configuration (nuc/robots.py).')

    # The joint-trajectory controller the home service moves the arm through; the hardware launch spawns
    # only the broadcasters. Activated inside the realtime loop where the distro allows it: requested from
    # the non-realtime thread, the switch to torque control collides with the read in flight and libfranka
    # refuses it while the controller still shows active.
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name=f'{robot.arm_controller}_spawner',
        output='screen',
        arguments=robots.spawner_arguments(
            robot.arm_controller,
            robots.JTC_TYPE,
            param_file(robot.arm_controller_mock_params if on_mock else robot.arm_controller_params),
            switch_asap=robot.switch_asap,
            timeout_s=CONTROLLER_MANAGER_TIMEOUT_S,
        ),
    )

    actions = [
        LogInfo(msg=f'[bringup] robot={robot.name} ({robot.arm_id} at {robot_ip}); TCP {calib.describe()}'),
        hardware(robot, robot_ip, load_gripper, use_fake_hardware),
        # The frame the policy speaks in, published whenever the arm is, so Foxglove and
        # `tf2_echo <arm>_link0 <tcp>` check the calibration without the inference launch.
        static_tf(f'{robot.tcp_frame}_static_tf', calib.static_transform_publisher_args()),
        *flange_frames(robot, desk_end_effector, load_gripper == 'true'),
    ]

    if on_mock:
        # Nothing to check and nothing to push, so the spawner runs directly.
        return [*actions, arm_controller_spawner]

    # Tell the FCI what is bolted to the flange so its gravity compensation cancels the whole weight. This
    # is the ONLY window: the robot refuses SetLoad once a controller holds the arm, hence the ordering.
    set_load = ExecuteProcess(
        cmd=['python3', str(NUC_DIR / 'set_payload.py'), CHECK_TIMEOUT_S, '--calib', robot.calib],
        output='screen',
    )
    # SetLoad can make franka_robot_state_broadcaster's next update fail (up to ~0.5 s after it lands), and the
    # controller manager then deactivates it for good: no external wrench, no force in the demos. Re-activate it
    # once the payload has settled; a no-op when it survived.
    broadcaster = 'franka_robot_state_broadcaster'
    revive_broadcaster = ExecuteProcess(
        cmd=['bash', '-c', f'sleep 2; if ros2 control list_controllers | grep {broadcaster} | grep -q inactive; then '
             f'echo "[bringup] {broadcaster} went inactive after SetLoad; re-activating"; '
             f'ros2 control switch_controllers --activate {broadcaster}; '
             f'else echo "[bringup] {broadcaster} active"; fi'],
        output='screen',
    )  # fmt: skip
    actions.append(
        RegisterEventHandler(
            OnProcessExit(
                target_action=set_load,
                on_exit=gate(
                    [arm_controller_spawner, revive_broadcaster],
                    '\n!!! SetLoad FAILED (exit {code}) -- the payload did not configure.\n'
                    f'!!! NOT spawning {robot.arm_controller}: the impedance controller would fight gravity it\n'
                    '!!! does not know about, and setLoad cannot be retried once a controller holds the arm.\n'
                    '!!! The real reason is in the /service_server log, not the response above:\n'
                    '!!!   grep -i "command exception" $(ls -t ~/.ros/log/ros2_control_node_*.log | head -1)\n'
                    f'!!! See docs/calibration-instructions.md ("End-effector payload") and nuc/{robot.calib}.py.\n',
                ),
            )
        )
    )

    if not robot.check_ee:
        return [*actions, set_load]

    # The <arm>_hand_tcp published above is only right if it matches the F_T_EE the robot applies. Ahead of
    # set_payload: past a wrong frame the controller servos about a point 10 cm from where the policy
    # thinks the TCP is, and nothing says so.
    check_ee_frame = ExecuteProcess(
        cmd=['python3', str(NUC_DIR / 'check_ee_frame.py'), CHECK_TIMEOUT_S, '--calib', robot.calib,
             '--desk-end-effector', desk_end_effector],
        output='screen',
    )  # fmt: skip
    return [
        *actions,
        check_ee_frame,
        RegisterEventHandler(
            OnProcessExit(
                target_action=check_ee_frame,
                on_exit=gate(
                    [set_load],
                    f'\n!!! EE FRAME CHECK FAILED (exit {{code}}): the {robot.arm_id}_hand_tcp this launch publishes\n'
                    '!!! does not match the end effector configured in Desk. The streaming controller would servo\n'
                    '!!! about the wrong point. Relaunch with desk_end_effector:=none|hand to match Desk, or change\n'
                    f'!!! it in Desk. See nuc/{robot.calib}.py ("Where fer_hand_tcp comes from").\n',
                ),
            )
        ),
    ]


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
            # Both arms default false: the FER has no hand, and on the FR3 the gripper is driven by
            # inference.launch.py's drivers, which cannot share the libfranka connection with franka_gripper.
            # The flag also reaches `xacro hand:=`; flange_frames() republishes what the URDF loses.
            DeclareLaunchArgument(
                'load_gripper',
                default_value='false',
                description='Start franka_gripper and put a hand in the URDF.',
            ),
            DeclareLaunchArgument(
                'use_fake_hardware',
                default_value='false',
                description='ros2_control mock hardware instead of the robot: no FCI, no gates, no torques (FER only).',
            ),
            DeclareLaunchArgument(
                'desk_end_effector',
                default_value=hsa_calib.DESK_END_EFFECTOR,
                choices=sorted(hsa_calib.DESK_EE_TRANSFORMS),
                description='FER only: the end effector Desk is configured with; sets fer_link8 -> fer_hand_tcp '
                '(the frame franka reports O_T_EE in) and is verified against the robot at bringup.',
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
