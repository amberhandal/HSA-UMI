"""
Check what fer_bringup.launch.py starts under mock hardware versus the robot, and how it gates the arm.

The launch has three branches that can do damage by being wrong: the two bringup-time checks
(EE frame, payload) must run on the robot and be skipped on mock hardware, the arm controller must
come up only through those gates on the robot, and exactly one fer_hand_tcp transform must be
published, the one matching the Desk setting. Evaluating the conditions needs no ROS graph, no
hardware, and no station.
"""

import importlib.util
import pathlib
from types import SimpleNamespace

from launch import LaunchContext
from launch.substitutions import LaunchConfiguration
import pytest

_LAUNCH_FILE = pathlib.Path(__file__).resolve().parent / 'launch' / 'fer_bringup.launch.py'


def _module():
    spec = importlib.util.spec_from_file_location('fer_bringup_launch', _LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv_text(entity) -> str:
    """
    Return the plain-text parts of an action's command line, for identifying it without performing substitutions.

    A Node's package/executable resolve only at launch time, but its arguments (controller names,
    frame ids) and an ExecuteProcess's script path are literal strings already.
    """
    parts = []
    for part in entity.process_description.cmd:
        for sub in part:
            text = getattr(sub, 'text', None)
            if text is not None:
                parts.append(text)
    return ' '.join(parts)


def _started(use_fake_hardware: str, desk_end_effector: str = 'none') -> list[str]:
    """Return the argv text of every CONDITIONED top-level action that would start."""
    context = LaunchContext()
    context.launch_configurations['use_fake_hardware'] = use_fake_hardware
    context.launch_configurations['desk_end_effector'] = desk_end_effector
    started = []
    for entity in _module().generate_launch_description().entities:
        condition = getattr(entity, 'condition', None)
        if condition is None or not hasattr(entity, 'process_description'):
            continue
        if condition.evaluate(context):
            started.append(_argv_text(entity))
    return started


def test_robot_runs_the_ee_gate_but_no_direct_spawner():
    """On the robot the arm controller may only come up through check_ee -> set_load."""
    started = _started('false')
    assert any('check_ee_frame.py' in s and '--desk-end-effector' in s for s in started)
    assert not any('set_payload.py' in s for s in started), 'set_load is started by the EE gate, never directly'
    assert not any(s.startswith('fer_arm_controller') for s in started), 'the spawner is started by the payload gate'


def test_mock_hardware_skips_the_gates_and_spawns_the_arm_controller_directly():
    """Mock hardware has no param service and no robot state, so nothing may wait on them."""
    started = _started('true')
    assert not any('check_ee_frame.py' in s or 'set_payload.py' in s for s in started)
    assert any(s.startswith('fer_arm_controller') for s in started)


def test_arm_controller_spawner_switches_inside_the_realtime_loop():
    """Without --switch-asap the effort switch is refused by libfranka and the arm never enters torque mode."""
    spawner = next(s for s in _started('true') if s.startswith('fer_arm_controller'))
    assert '--switch-asap' in spawner


def test_franka_bringup_is_included_with_the_hardware_flag_forwarded():
    """
    The broadcasters come from the included hardware launch file, keyed on its own use_fake_hardware.

    It spawns joint_state_broadcaster always and franka_robot_state_broadcaster only on the robot, so
    this file must forward the flag, or mock hardware would wait forever on a broadcaster that cannot come up.
    """
    from launch.actions import IncludeLaunchDescription

    includes = [e for e in _module().generate_launch_description().entities if isinstance(e, IncludeLaunchDescription)]
    assert len(includes) == 1
    forwarded = {name for name, _value in includes[0].launch_arguments}
    assert {'robot_ip', 'arm_id', 'load_gripper', 'use_fake_hardware'} <= forwarded


@pytest.mark.parametrize('setting', ['none', 'hand'])
def test_exactly_one_fer_hand_tcp_transform_is_published(setting):
    """Two would make TF pick one arbitrarily; none would stop the impedance controller activating."""
    started = _started('false', setting)
    ee = [s for s in started if '--child-frame-id fer_hand_tcp' in s]
    assert len(ee) == 1
    # The 'hand' transform carries the 0.1034 m offset; 'none' is all zeros.
    assert ('0.1034' in ee[0]) == (setting == 'hand')


def test_gates_start_the_next_step_only_on_exit_zero():
    """A rejected payload or a mismatched frame must abort the launch, and loudly (non-zero exit)."""
    module = _module()
    handler = module.gate(['NEXT'], 'failed with {code}')
    assert handler(SimpleNamespace(returncode=0), None) == ['NEXT']
    with pytest.raises(RuntimeError, match='failed with 3'):
        handler(SimpleNamespace(returncode=3), None)


@pytest.mark.parametrize('code', [130, -2, -15])
def test_gates_stay_quiet_when_the_gated_process_was_interrupted(code):
    """A launch already shutting down must not get a second, false diagnosis from the gate."""
    handler = _module().gate(['NEXT'], 'failed with {code}')
    assert handler(SimpleNamespace(returncode=code), None) == []


@pytest.mark.parametrize(
    ('use_fake_hardware', 'expected'),
    [('false', 'fer_arm_controller.yaml'), ('true', 'fer_arm_controller.mock.yaml')],
)
def test_controller_manager_yaml_follows_the_hardware(use_fake_hardware, expected):
    """The effort variant would spin mock hardware without moving it; the position variant is not for the robot."""
    context = LaunchContext()
    context.launch_configurations['use_fake_hardware'] = use_fake_hardware
    expression = _module().controller_manager_yaml(LaunchConfiguration('use_fake_hardware'))
    assert context.perform_substitution(expression) == expected


@pytest.mark.parametrize(
    ('yaml_name', 'command_interface'),
    [('fer_arm_controller.yaml', 'effort'), ('fer_arm_controller.mock.yaml', 'position')],
)
def test_arm_controller_yamls_carry_the_type_and_the_seven_joints(yaml_name, command_interface):
    """
    The spawner reads `type` from the parameter file and nowhere else on Jazzy/Kilted.

    NU-MSR's own files keep it under controller_manager, which the spawner does not look at, so a
    copy that lost the key would fail to load with "type param was not defined". The command
    interface is pinned too: effort is what the robot and the controller switch need, position is
    the only thing mock hardware tracks.
    """
    import yaml

    doc = yaml.safe_load((pathlib.Path(__file__).resolve().parent / 'config' / yaml_name).read_text())
    params = doc['fer_arm_controller']['ros__parameters']
    assert params['type'] == 'joint_trajectory_controller/JointTrajectoryController'
    assert params['command_interfaces'] == [command_interface]
    assert params['joints'] == [f'fer_joint{i}' for i in range(1, 8)]


def test_impedance_controller_yaml_carries_its_type():
    """Same rule for the streaming controller's file, spawned by fer_inference.launch.py."""
    import yaml

    doc = yaml.safe_load((pathlib.Path(__file__).resolve().parent / 'config' / 'hsa_controllers.yaml').read_text())
    params = doc['polyumi_cartesian_impedance_controller']['ros__parameters']
    assert params['type'] == 'franka_streaming_impedance_controller/CartesianImpedanceController'
    assert params['tcp_frame'] == 'hsa_tcp' and params['base_frame'] == 'fer_link0' and params['arm_id'] == 'fer'


def test_controller_manager_yaml_disables_overrun_management():
    """
    The manager must not re-time its loop after a late cycle, or half the torque commands miss the robot.

    ros2_control issue #2529: franka_hardware's read() blocks until the robot's next packet and paces
    the loop by itself.
    """
    import yaml

    doc = yaml.safe_load((pathlib.Path(__file__).resolve().parent / 'config' / 'controller_manager.yaml').read_text())
    params = doc['controller_manager']['ros__parameters']
    assert params['overruns.manage'] is False
    assert params['update_rate'] == 1000
    assert params['joint_state_broadcaster']['type'] == 'joint_state_broadcaster/JointStateBroadcaster'
    assert (
        params['franka_robot_state_broadcaster']['type'] == 'franka_robot_state_broadcaster/FrankaRobotStateBroadcaster'
    )


def test_hardware_launch_is_ours_and_takes_our_yaml():
    """The include must point at nuc/launch/fer_hardware.launch.py, whose manager yaml is ours by default."""
    from launch.actions import IncludeLaunchDescription

    includes = [e for e in _module().generate_launch_description().entities if isinstance(e, IncludeLaunchDescription)]
    # The source resolves its path only when the included description is loaded; load it, which
    # also proves the hardware launch file parses.
    source = includes[0].launch_description_source
    included = source.get_launch_description(LaunchContext())
    assert 'fer_hardware.launch.py' in source.location
    assert included.entities, 'the hardware launch produced no entities'
    hw = pathlib.Path(__file__).resolve().parent / 'launch' / 'fer_hardware.launch.py'
    assert 'controllers_yaml' in hw.read_text() and 'controller_manager.yaml' in hw.read_text()
