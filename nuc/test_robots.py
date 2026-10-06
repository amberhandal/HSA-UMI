"""
Check the robot table: every profile points at files that exist, and the laptop side agrees on the frames.

nuc/robots.py and ros2_ws/src/hsa_ros2/config/robots/<name>.yaml are two statements of the same frames
(the arm box has no HSA-UMI workspace to read the yaml from); a mismatch would leave the policy client
looking up a TCP that nobody publishes.
"""

import importlib
import pathlib
import sys

import pytest
import yaml

_NUC = pathlib.Path(__file__).resolve().parent
_HSA_ROBOTS = _NUC.parent / 'ros2_ws' / 'src' / 'hsa_ros2' / 'config' / 'robots'
sys.path.insert(0, str(_NUC))

import robots  # noqa: E402


@pytest.mark.parametrize('name', sorted(robots.ROBOTS))
def test_profile_files_exist(name):
    """A typo in a file name would only surface on the arm box, at bringup."""
    robot = robots.ROBOTS[name]
    assert robot.name == name
    assert (_NUC / robot.home_service).is_file()
    assert (_NUC / 'config' / robot.servo_params).is_file()
    for params in (robot.arm_controller_params, robot.arm_controller_mock_params):
        if params and not params.startswith('package://'):
            assert (_NUC / 'config' / params).is_file()


@pytest.mark.parametrize('name', sorted(robots.ROBOTS))
def test_calibration_module_publishes_the_profiles_tcp(name):
    """The TCP static transform and the profile name the same frame, off the same arm."""
    robot = robots.ROBOTS[name]
    calib = importlib.import_module(robot.calib)
    args = calib.static_transform_publisher_args()
    assert args[args.index('--child-frame-id') + 1] == robot.tcp_frame
    assert args[args.index('--frame-id') + 1].startswith(f'{robot.arm_id}_')


@pytest.mark.parametrize('name', sorted(robots.ROBOTS))
def test_servo_yaml_matches_the_profile(name):
    """The streaming controller converts onto tcp_frame in base_frame; both must be this robot's."""
    robot = robots.ROBOTS[name]
    doc = yaml.safe_load((_NUC / 'config' / robot.servo_params).read_text())
    params = doc[robots.SERVO_CONTROLLER]['ros__parameters']
    assert params['arm_id'] == robot.arm_id
    assert params['base_frame'] == robot.base_frame
    assert params['tcp_frame'] == robot.tcp_frame


@pytest.mark.parametrize('name', sorted(robots.ROBOTS))
def test_laptop_robot_yaml_agrees(name):
    """Every hsa_ros2 node names the base and TCP frames from config/robots/<name>.yaml."""
    robot = robots.ROBOTS[name]
    params = yaml.safe_load((_HSA_ROBOTS / f'{name}.yaml').read_text())['/**']['ros__parameters']
    assert params['base_frame'] == robot.base_frame
    assert params['tcp_frame'] == params['eef_frame'] == robot.tcp_frame


def test_every_laptop_robot_yaml_has_a_profile():
    """A yaml without a profile is a robot the laptop can select and the arm box cannot."""
    assert {p.stem for p in _HSA_ROBOTS.glob('*.yaml')} == set(robots.ROBOTS)


def test_get_reads_the_environment_and_refuses_typos(monkeypatch):
    """robot:= defaults to $HSA_ROBOT, then fer; an unknown name is an error, not a fallback."""
    monkeypatch.delenv(robots.ROBOT_ENV, raising=False)
    assert robots.get('').name == 'fer'
    monkeypatch.setenv(robots.ROBOT_ENV, 'fr3')
    assert robots.get(None).name == 'fr3'
    assert robots.get('fer').name == 'fer'
    with pytest.raises(ValueError, match='fr3'):
        robots.get('panda')


@pytest.mark.parametrize(('distro', 'flag'), [('humble', True), ('jazzy', False), ('kilted', False)])
def test_spawner_type_flag_follows_the_distro(distro, flag):
    """Humble needs -t; Jazzy and later reject it and read the type from the parameter file."""
    args = robots.spawner_arguments('c', 'pkg/Type', 'f.yaml', timeout_s='9', ros_distro=distro)
    assert ('-t' in args) is flag
    assert args[0] == 'c' and args[-2:] == ['--controller-manager-timeout', '9']
