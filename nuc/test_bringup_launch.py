"""
Check what bringup.launch.py starts for each robot, under mock hardware and on the robot, and how it gates the arm.

The branches that can do damage by being wrong: the bringup gates (EE frame, payload) must run on the robot
and be skipped on mock hardware, the arm controller must come up only through them on the robot, and the
fixed frames the URDF loses must be republished exactly once. Evaluating launch_setup needs no ROS graph,
no hardware and no arm box.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

from launch import LaunchContext
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch_ros.actions import Node
import pytest

_NUC = pathlib.Path(__file__).resolve().parent
_LAUNCH_FILE = _NUC / 'launch' / 'bringup.launch.py'


def _module():
    spec = importlib.util.spec_from_file_location('bringup_launch', _LAUNCH_FILE)
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


def _node_args(node: Node) -> str:
    return ' '.join(str(a) for a in node._Node__arguments)


@pytest.fixture
def actions(monkeypatch):
    """Return a function building launch_setup's actions for the given arguments and ROS distro."""

    def build(robot='fer', use_fake_hardware='false', desk_end_effector='none', load_gripper='false', distro='kilted'):
        monkeypatch.setenv('ROS_DISTRO', distro)
        context = LaunchContext()
        context.launch_configurations.update(
            robot=robot,
            robot_ip='',
            use_fake_hardware=use_fake_hardware,
            desk_end_effector=desk_end_effector,
            load_gripper=load_gripper,
        )
        return _module().launch_setup(context)

    return build


def _started(acts) -> list[str]:
    """Argv text of every process started directly (not by a gate)."""
    return [_argv_text(a) for a in acts if hasattr(a, 'process_description')]


def _gated(acts) -> dict[str, list[str]]:
    """Map each gated process (argv text) to the argv text of what its exit 0 starts."""
    out = {}
    processes = [a for a in acts if hasattr(a, 'process_description')]
    for register in (a for a in acts if isinstance(a, RegisterEventHandler)):
        handler = register.event_handler
        matcher = handler._OnActionEventBase__action_matcher
        (target,) = [p for p in processes if (matcher(p) if callable(matcher) else matcher is p)]
        started = handler.handle(SimpleNamespace(returncode=0, action=target), LaunchContext())
        out[_argv_text(target)] = [_argv_text(a) for a in started]
    return out


def test_fer_robot_chains_check_ee_then_payload_then_spawner(actions):
    """On the robot the arm controller may only come up through check_ee -> set_load."""
    acts = actions('fer')
    started = _started(acts)
    assert [s for s in started if 'check_ee_frame.py' in s and '--calib hsa_calib' in s]
    assert not any('set_payload.py' in s for s in started), 'set_load is started by the EE gate, never directly'
    assert not any(s.startswith('fer_arm_controller') for s in started)
    # set_load is only reachable through check_ee's handler, so walk the chain by hand.
    (register_check,) = [a for a in acts if isinstance(a, RegisterEventHandler)][-1:]
    (check,) = [a for a in acts if hasattr(a, 'process_description') and 'check_ee_frame.py' in _argv_text(a)]
    (set_load,) = register_check.event_handler.handle(SimpleNamespace(returncode=0, action=check), LaunchContext())
    assert 'set_payload.py' in _argv_text(set_load) and '--calib hsa_calib' in _argv_text(set_load)
    (register_load,) = [a for a in acts if isinstance(a, RegisterEventHandler)][:1]
    (spawner,) = register_load.event_handler.handle(SimpleNamespace(returncode=0, action=set_load), LaunchContext())
    assert _argv_text(spawner).startswith('fer_arm_controller')
    assert '--switch-asap' in _argv_text(spawner)


def test_fr3_robot_has_no_ee_check_and_gates_the_spawner_on_the_payload(actions):
    """The FR3 keeps its hand frames from tcp_calib, so only the payload gate applies."""
    acts = actions('fr3', distro='humble')
    started = _started(acts)
    assert any('set_payload.py' in s and '--calib tcp_calib' in s for s in started)
    assert not any('check_ee_frame.py' in s for s in started)
    ((spawner,),) = _gated(acts).values()
    assert spawner.startswith('fr3_arm_controller -t joint_trajectory_controller/JointTrajectoryController')
    assert '--switch-asap' not in spawner, "Humble's spawner has no --switch-asap"


def test_mock_hardware_skips_the_gates_and_spawns_the_arm_controller_directly(actions):
    """Mock hardware has no param service and no robot state, so nothing may wait on them."""
    started = _started(actions('fer', use_fake_hardware='true'))
    assert not any('check_ee_frame.py' in s or 'set_payload.py' in s for s in started)
    (spawner,) = [s for s in started if s.startswith('fer_arm_controller')]
    assert '-t' not in spawner.split(), 'Jazzy/Kilted spawners have no -t; the type lives in the yaml'


def test_fr3_refuses_mock_hardware(actions):
    """There is no position-commanded FR3 controller file, so mock hardware would never move."""
    with pytest.raises(RuntimeError, match='mock'):
        actions('fr3', use_fake_hardware='true')


def test_unknown_robot_is_refused(actions):
    """A typo must not fall back to some other arm."""
    with pytest.raises(ValueError, match='fer'):
        actions('panda')


@pytest.mark.parametrize(('robot', 'ours'), [('fer', True), ('fr3', False)])
def test_hardware_launch_per_robot(actions, robot, ours):
    """The FER takes our copy (overrun handling, mock flag); the FR3 the franka_bringup on its NUC."""
    (include,) = [a for a in actions(robot) if isinstance(a, IncludeLaunchDescription)]
    args = dict(include.launch_arguments)
    assert args['arm_id'] == robot
    if ours:
        source = include.launch_description_source
        # The location resolves when the description is loaded, which also proves the file parses.
        assert source.get_launch_description(LaunchContext()).entities
        assert 'fer_hardware.launch.py' in source.location
        assert args['use_fake_hardware'] == 'false'
    else:
        assert 'use_fake_hardware' not in args


@pytest.mark.parametrize('setting', ['none', 'hand'])
def test_fer_publishes_exactly_one_hand_tcp_transform(actions, setting):
    """Two would make TF pick one arbitrarily; none would stop the impedance controller activating."""
    args = [_node_args(n) for n in actions('fer', desk_end_effector=setting) if isinstance(n, Node)]
    ee = [a for a in args if '--child-frame-id fer_hand_tcp' in a]
    assert len(ee) == 1
    assert ('0.1034' in ee[0]) == (setting == 'hand')


@pytest.mark.parametrize(('robot', 'tcp'), [('fer', 'hsa_tcp'), ('fr3', 'polyumi_tcp')])
def test_tcp_frame_is_published(actions, robot, tcp):
    """The frame the policy observes, from the robot's calibration module."""
    args = [_node_args(n) for n in actions(robot) if isinstance(n, Node)]
    assert any(f'--child-frame-id {tcp}' in a for a in args)


def test_fr3_republishes_the_hand_only_without_franka_gripper(actions):
    """With load_gripper the URDF has the hand; publishing it again would give TF two parents."""

    def hand(load_gripper):
        args = [_node_args(n) for n in actions('fr3', load_gripper=load_gripper) if isinstance(n, Node)]
        return [a for a in args if 'fr3_hand_tcp' in a]

    assert hand('false') and not hand('true')


def test_gates_start_the_next_step_only_on_exit_zero():
    """A rejected payload or a mismatched frame must abort the launch, and loudly (non-zero exit)."""
    handler = _module().gate(['NEXT'], 'failed with {code}')
    assert handler(SimpleNamespace(returncode=0), None) == ['NEXT']
    with pytest.raises(RuntimeError, match='failed with 3'):
        handler(SimpleNamespace(returncode=3), None)


@pytest.mark.parametrize('code', [130, -2, -15])
def test_gates_stay_quiet_when_the_gated_process_was_interrupted(code):
    """A launch already shutting down must not get a second, false diagnosis from the gate."""
    assert _module().gate(['NEXT'], 'failed with {code}')(SimpleNamespace(returncode=code), None) == []


def test_package_param_files_resolve_through_find_package_share():
    """The FR3's arm-controller file lives in franka_fr3_moveit_config on the NUC, not in nuc/config."""
    from launch.substitutions import PathJoinSubstitution

    module = _module()
    assert isinstance(module.param_file('package://pkg/config/x.yaml'), PathJoinSubstitution)
    assert module.param_file('x.yaml') == str(_NUC / 'config' / 'x.yaml')


@pytest.mark.parametrize(
    ('yaml_name', 'command_interface'),
    [('fer_arm_controller.yaml', 'effort'), ('fer_arm_controller.mock.yaml', 'position')],
)
def test_arm_controller_yamls_carry_the_type_and_the_seven_joints(yaml_name, command_interface):
    """
    The Jazzy/Kilted spawner reads `type` from the parameter file and nowhere else.

    The command interface is pinned too: effort is what the robot and the controller switch need,
    position is the only thing mock hardware tracks.
    """
    import yaml

    params = yaml.safe_load((_NUC / 'config' / yaml_name).read_text())['fer_arm_controller']['ros__parameters']
    assert params['type'] == 'joint_trajectory_controller/JointTrajectoryController'
    assert params['command_interfaces'] == [command_interface]
    assert params['joints'] == [f'fer_joint{i}' for i in range(1, 8)]


def test_controller_manager_yaml_disables_overrun_management():
    """
    The manager must not re-time its loop after a late cycle, or half the torque commands miss the robot.

    ros2_control issue #2529: franka_hardware's read() blocks until the robot's next packet and paces
    the loop by itself.
    """
    import yaml

    doc = yaml.safe_load((_NUC / 'config' / 'controller_manager.yaml').read_text())
    params = doc['controller_manager']['ros__parameters']
    assert params['overruns.manage'] is False
    assert params['update_rate'] == 1000
    assert params['joint_state_broadcaster']['type'] == 'joint_state_broadcaster/JointStateBroadcaster'


def test_generate_launch_description_defers_to_launch_setup():
    """Every robot-dependent action is built at launch time, once `robot` is known."""
    from launch.actions import OpaqueFunction

    entities = _module().generate_launch_description().entities
    assert sum(isinstance(e, OpaqueFunction) for e in entities) == 1
