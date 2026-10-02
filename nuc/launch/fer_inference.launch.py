"""
Launch the station-side inference stack: the home service plus the streaming controller.

Run on the station, after fer_bringup.launch.py is up:

    ros2 launch nuc/launch/fer_inference.launch.py                    # loaded, nothing moves
    ros2 launch nuc/launch/fer_inference.launch.py execute_arm:=true  # servo drives the arm

The streaming Cartesian impedance controller is loaded inactive and only ACTIVATED with `execute_arm:=true`;
otherwise the arm stays on fer_arm_controller. Separate from fer_bringup.launch.py so that bringup, the
crash-prone, FCI-gated piece, restarts alone; these nodes tolerate starting before their peers, but
fer_arm_controller must already exist or /polyumi/home has nothing to drive. See docs/hsa-arm-inference.md.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'
TRAJECTORY_CONTROLLER = 'fer_arm_controller'

# fer_home_service is a standalone script, not an installed ament package (it runs from a plain
# clone on the station, which has no HSA-UMI workspace), so it is ExecuteProcess by path.
NUC_DIR = Path(__file__).resolve().parent.parent


def switch_to_servo(execute_arm) -> ExecuteProcess:
    """
    Build the controller switch that hands the arm to the servo, gated on `execute_arm`.

    `ros2 service call`, NOT `ros2 control switch_controllers`: ros2controlcli goes through the ros2cli
    daemon, and one poisoned daemon fails this with `RuntimeError: !rclpy.ok()` while every other pane
    looks healthy. strictness 2 = STRICT: fail rather than leave two controllers claiming the same interfaces.
    """
    return ExecuteProcess(
        cmd=[
            'ros2',
            'service',
            'call',
            '/controller_manager/switch_controller',
            'controller_manager_msgs/srv/SwitchController',
            # activate_asap: switch inside the realtime loop, between read() and write(); from the
            # non-realtime thread libfranka refuses the mode change (see fer_bringup.launch.py).
            f"{{activate_controllers: ['{SERVO_CONTROLLER}'], "
            f"deactivate_controllers: ['{TRAJECTORY_CONTROLLER}'], strictness: 2, activate_asap: true}}",
        ],
        output='screen',
        condition=IfCondition(PythonExpression(["'", execute_arm, "' == 'true'"])),
    )


def generate_launch_description():
    """Start the home service and load the streaming controller; activate it only on execute_arm."""
    execute_arm = LaunchConfiguration('execute_arm')

    # Hoisted so the event handler can name it: the switch has to wait for the spawner to finish,
    # and launch offers no ordering guarantee otherwise.
    impedance_spawner = Node(
        package='controller_manager',
        executable='spawner',
        name='polyumi_impedance_controller_spawner',
        output='screen',
        arguments=[
            SERVO_CONTROLLER,
            # The file carries the controller's `type` too: Jazzy's and Kilted's spawner has no -t
            # flag and reads it from there (see nuc/config/hsa_controllers.yaml).
            '--param-file',
            str(NUC_DIR / 'config' / 'hsa_controllers.yaml'),
            # Inactive in both modes. Activating means claiming the effort interfaces
            # fer_arm_controller already holds, which a spawner cannot do -- that takes the switch.
            '--inactive',
            '--controller-manager-timeout',
            '30',
        ],
    )

    return LaunchDescription(
        [
            # Defaults false: launching this file must never move the robot on its own. Torque
            # control starts the moment the controller activates, so the switch is gated on it.
            DeclareLaunchArgument(
                'home_pose_file',
                default_value='~/.ros/polyumi_home_pose.yaml',
                description='Where /polyumi/set_home persists the taught pose and /polyumi/home reads it. '
                "Mock-hardware runs pass their own file so they never overwrite the real arm's.",
            ),
            DeclareLaunchArgument(
                'execute_arm',
                default_value='false',
                description='Activate the streaming Cartesian impedance controller (MOVES THE ARM on the next chunk).',
            ),
            # Always started: it owns /polyumi/home and /polyumi/set_home, and homing borrows the arm
            # back from the servo by itself. The pose file is an argument so mock runs never touch the real one.
            ExecuteProcess(
                cmd=[
                    'python3',
                    str(NUC_DIR / 'fer_home_service.py'),
                    '--ros-args',
                    '-p',
                    ['home_pose_file:=', LaunchConfiguration('home_pose_file')],
                ],
                output='screen',
            ),  # fmt: skip
            impedance_spawner,
            # Hand the arm to the servo once the spawner has actually loaded it; switching to a
            # controller that is not loaded yet just fails.
            RegisterEventHandler(
                OnProcessExit(target_action=impedance_spawner, on_exit=[switch_to_servo(execute_arm)])
            ),
        ]
    )
