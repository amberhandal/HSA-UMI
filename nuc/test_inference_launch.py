"""
Check that inference.launch.py loads the servo inactive and hands the arm over ONLY on execute_arm.

It must also start the right homing and gripper pieces for each robot.

The switch is the one thing here that can move the arm: once the streaming controller is active, the next
chunk on /polyumi/target_poses_traj drives torques. The two FR3 grippers are mutually exclusive hardware.
Evaluating launch_setup needs no ROS graph, no hardware and no arm box.
"""

import importlib.util
import pathlib

from launch import LaunchContext
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch_ros.actions import Node
import pytest

_LAUNCH_FILE = pathlib.Path(__file__).resolve().parent / 'launch' / 'inference.launch.py'

HAND = 'franka_hand_node'
FAULHABER = 'faulhaber_gripper.launch.xml'


def _module():
    spec = importlib.util.spec_from_file_location('inference_launch', _LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv_text(entity) -> str:
    """Return the literal parts of an action's command line (package/executable resolve only at launch time)."""
    parts = []
    for part in entity.process_description.cmd:
        for sub in part:
            text = getattr(sub, 'text', None)
            if text is not None:
                parts.append(text)
    return ' '.join(parts)


@pytest.fixture
def actions(monkeypatch):
    """Return a function building launch_setup's actions for the given arguments and ROS distro."""

    def build(robot='fer', execute_arm='false', execute_gripper='false', gripper='faulhaber', home_pose_file='',
              distro='kilted'):  # fmt: skip
        monkeypatch.setenv('ROS_DISTRO', distro)
        context = LaunchContext()
        context.launch_configurations.update(
            robot=robot,
            robot_ip='',
            home_pose_file=home_pose_file,
            execute_arm=execute_arm,
            execute_gripper=execute_gripper,
            gripper=gripper,
            max_acceleration='1.5',
        )
        return _module().launch_setup(context)

    return build


def _switches(acts) -> list[str]:
    out = []
    for register in (a for a in acts if isinstance(a, RegisterEventHandler)):
        _description, emitted = register.event_handler.describe()
        out += [_argv_text(a) for a in emitted]
    return out


@pytest.mark.parametrize('robot', ['fer', 'fr3'])
def test_default_never_switches_to_the_servo(actions, robot):
    """The default (execute_arm false) must never activate the servo."""
    assert _switches(actions(robot)) == []


@pytest.mark.parametrize(('robot', 'arm_controller', 'asap'), [('fer', 'fer_arm_controller', True),
                                                               ('fr3', 'fr3_arm_controller', False)])  # fmt: skip
def test_execute_arm_switches_after_the_spawner(actions, robot, arm_controller, asap):
    """Switching before the spawner has loaded the controller just fails, so it hangs on the spawner's exit."""
    (switch,) = _switches(actions(robot, execute_arm='true'))
    assert 'switch_controller' in switch and 'polyumi_cartesian_impedance_controller' in switch
    assert arm_controller in switch and 'strictness: 2' in switch
    assert ('activate_asap: true' in switch) is asap


SERVO_CASES = [('fer', 'kilted', 'hsa_controllers.yaml', False), ('fr3', 'humble', 'polyumi_controllers.yaml', True)]
HOME_CASES = [
    ('fer', 'fer_home_service.py', 'polyumi_home_pose.yaml'),
    ('fr3', 'fr3_home_service.py', 'hsa_home_pose.yaml'),
]


@pytest.mark.parametrize(('robot', 'distro', 'yaml', 'type_flag'), SERVO_CASES)
def test_servo_is_spawned_inactive_with_the_robots_yaml(actions, robot, distro, yaml, type_flag):
    """Loaded, never activated by the spawner; -t only where the spawner has it (Humble)."""
    spawners = [a for a in actions(robot, distro=distro) if isinstance(a, Node) and 'spawner' in str(a.node_executable)]
    (spawner,) = spawners
    joined = ' '.join(str(a) for a in spawner._Node__arguments)
    assert '--inactive' in joined and yaml in joined
    assert ('-t' in joined.split()) is type_flag


@pytest.mark.parametrize(('robot', 'service', 'pose_file'), HOME_CASES)
def test_home_service_is_always_started_with_the_robots_pose_file(actions, robot, service, pose_file):
    """/polyumi/home must exist whether or not the servo may move; the FR3's pose file is not CRB's."""
    cmds = [_argv_text(a) for a in actions(robot) if isinstance(a, ExecuteProcess) and not isinstance(a, Node)]
    (home,) = [c for c in cmds if service in c]
    assert f'home_pose_file:=~/.ros/{pose_file}' in home


def test_home_pose_file_argument_wins(actions):
    """Mock runs pass their own file so they never overwrite the real arm's."""
    cmds = [_argv_text(a) for a in actions('fer', home_pose_file='/tmp/x.yaml') if isinstance(a, ExecuteProcess)]
    assert any('home_pose_file:=/tmp/x.yaml' in c for c in cmds)


def _location(source) -> str:
    """Return an include's launch file location as text, without resolving package paths."""
    location = source._LaunchDescriptionSource__location
    parts = location if isinstance(location, list) else [location]
    return ' '.join(getattr(p, 'describe', lambda p=p: str(p))() for p in parts)


@pytest.mark.parametrize(('robot', 'move_group'), [('fer', False), ('fr3', True)])
def test_move_group_only_for_moveit_homing(actions, robot, move_group):
    """The FER homes straight through its trajectory controller; the FR3 plans through move_group."""
    includes = [
        _location(a.launch_description_source) for a in actions(robot) if isinstance(a, IncludeLaunchDescription)
    ]
    assert any('fr3_move_group.launch.py' in i for i in includes) is move_group


def _drivers(acts) -> set:
    """Which gripper drivers these actions start: the hand is a Node, the FAULHABER an include."""
    started = set()
    for a in acts:
        if isinstance(a, Node) and HAND in str(a.node_executable):
            started.add(HAND)
        if isinstance(a, IncludeLaunchDescription) and FAULHABER in _location(a.launch_description_source):
            started.add(FAULHABER)
    return started


@pytest.mark.parametrize(
    ('gripper', 'execute_gripper', 'expected'),
    [
        # The defaults. Nothing may move on a bare `ros2 launch`.
        ('hand', 'false', set()),
        ('faulhaber', 'false', set()),
        ('none', 'false', set()),
        # execute_gripper alone is not enough; `gripper` says which device.
        ('hand', 'true', {HAND}),
        ('faulhaber', 'true', {FAULHABER}),
        ('none', 'true', set()),
    ],
)
def test_fr3_starts_exactly_the_selected_gripper(actions, gripper, execute_gripper, expected):
    """Never both drivers at once: they would fight over one set of fingers."""
    assert _drivers(actions('fr3', gripper=gripper, execute_gripper=execute_gripper)) == expected


def test_fer_never_starts_a_gripper_driver(actions):
    """The FER has no gripper; a stray flag must not start a driver that grabs nothing."""
    assert _drivers(actions('fer', gripper='hand', execute_gripper='true')) == set()

