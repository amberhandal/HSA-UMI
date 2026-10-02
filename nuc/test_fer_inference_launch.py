"""
Check that fer_inference.launch.py loads the servo inactive and hands the arm over ONLY on execute_arm.

The switch is the one thing in this launch file that can move the arm: once the streaming
controller is active, the next chunk on /polyumi/target_poses_traj drives torques. Evaluating the
condition needs no ROS graph, no hardware, and no station.
"""

import importlib.util
import pathlib

from launch import LaunchContext
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node
import pytest

_LAUNCH_FILE = pathlib.Path(__file__).resolve().parent / 'launch' / 'fer_inference.launch.py'


def _module():
    spec = importlib.util.spec_from_file_location('fer_inference_launch', _LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entities():
    return _module().generate_launch_description().entities


def _handler_actions(register: RegisterEventHandler) -> list:
    """Return the actions an OnProcessExit handler will emit, reached through launch's describe()."""
    _description, actions = register.event_handler.describe()
    return list(actions)


def _argv_text(entity) -> str:
    """Return the literal parts of an action's command line (package/executable resolve only at launch time)."""
    parts = []
    for part in entity.process_description.cmd:
        for sub in part:
            text = getattr(sub, 'text', None)
            if text is not None:
                parts.append(text)
    return ' '.join(parts)


@pytest.mark.parametrize(('execute_arm', 'expected'), [('false', False), ('true', True)])
def test_switch_to_servo_is_gated_on_execute_arm(execute_arm, expected):
    """The default (`false`) must never activate the servo; only an explicit `true` may."""
    module = _module()
    context = LaunchContext()
    context.launch_configurations['execute_arm'] = execute_arm

    switch = module.switch_to_servo(module.LaunchConfiguration('execute_arm'))

    assert switch.condition.evaluate(context) is expected


def test_switch_is_registered_on_the_spawners_exit_and_names_both_controllers():
    """Switching before the spawner has loaded the controller just fails, so the ordering is load-bearing."""
    registers = [e for e in _entities() if isinstance(e, RegisterEventHandler)]
    assert len(registers) == 1, 'exactly one event handler: the switch on the spawner exit'

    actions = _handler_actions(registers[0])
    assert len(actions) == 1 and isinstance(actions[0], ExecuteProcess)
    cmd = _argv_text(actions[0])
    assert 'switch_controller' in cmd
    assert 'polyumi_cartesian_impedance_controller' in cmd
    assert 'fer_arm_controller' in cmd
    assert 'strictness: 2' in cmd
    assert 'activate_asap: true' in cmd, 'the mode switch must run inside the realtime loop'


def test_servo_is_spawned_inactive_with_the_hsa_yaml():
    """Loaded, never activated by the spawner: activation is the switch's job, gated above."""
    spawners = [e for e in _entities() if isinstance(e, Node) and 'spawner' in str(e.node_executable)]
    assert len(spawners) == 1

    joined = _argv_text(spawners[0])
    assert '--inactive' in joined
    assert 'hsa_controllers.yaml' in joined
    assert '-t' not in joined.split(), 'Jazzy/Kilted spawners have no -t; the type lives in the yaml'


def test_home_service_is_always_started():
    """/polyumi/home must exist whether or not the servo is allowed to move."""
    processes = [e for e in _entities() if isinstance(e, ExecuteProcess) and not isinstance(e, Node)]
    cmds = [_argv_text(proc) for proc in processes]

    assert any('fer_home_service.py' in c for c in cmds), cmds
    assert any('home_pose_file:=' in c for c in cmds), (
        'the pose file must be a launch argument (mock runs use their own)'
    )
