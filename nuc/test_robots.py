"""
Check the robot table: every profile points at files that exist, and the laptop side agrees on the frames.

nuc/robots.py and ros2_ws/src/hsa_ros2/config/robots/<frames>.yaml are two statements of the same frames
(the arm box has no HSA-UMI workspace to read the yaml from); a mismatch would leave the policy client
looking up a TCP that nobody publishes. A profile is an arm plus what is on its flange, so every check runs
over each arm with and without an HSA tool.
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

#: (robot, HSA_TOOL): each arm with its own end effector, and with each HSA tool.
PROFILES = [(name, tool) for name in sorted(robots.ROBOTS) for tool in ('', *robots.HSA_TOOLS)]


@pytest.fixture
def profile(hsa_env):
    """Return a function exporting the profile's environment, as station_env.sh would, and returning it."""

    def build(name, tool):
        hsa_env(name, tool)
        return robots.get(name, tool)

    return build


@pytest.mark.parametrize(('name', 'tool'), PROFILES)
def test_profile_files_exist(profile, name, tool):
    """A typo in a file name would only surface on the arm box, at bringup."""
    robot = profile(name, tool)
    assert robot.name == name
    assert (_NUC / robot.home_service).is_file()
    assert (_NUC / 'config' / robot.servo_params).is_file()
    for params in (robot.arm_controller_params, robot.arm_controller_mock_params):
        if params and not params.startswith('package://'):
            assert (_NUC / 'config' / params).is_file()


@pytest.mark.parametrize(('name', 'tool'), PROFILES)
def test_calibration_module_publishes_the_profiles_tcp(profile, name, tool):
    """The TCP static transform and the profile name the same frame, off the same arm."""
    robot = profile(name, tool)
    calib = importlib.import_module(robot.calib)
    args = calib.static_transform_publisher_args()
    assert args[args.index('--child-frame-id') + 1] == robot.tcp_frame
    assert args[args.index('--frame-id') + 1].startswith(f'{robot.arm_id}_')


@pytest.mark.parametrize(('name', 'tool'), PROFILES)
def test_servo_yaml_matches_the_profile(profile, name, tool):
    """The streaming controller converts onto tcp_frame in base_frame; both must be this robot's."""
    robot = profile(name, tool)
    doc = yaml.safe_load((_NUC / 'config' / robot.servo_params).read_text())
    params = doc[robots.SERVO_CONTROLLER]['ros__parameters']
    assert params['arm_id'] == robot.arm_id
    assert params['base_frame'] == robot.base_frame
    assert params['tcp_frame'] == robot.tcp_frame


@pytest.mark.parametrize(('name', 'tool'), PROFILES)
def test_laptop_robot_yaml_agrees(profile, name, tool):
    """Every hsa_ros2 node names the base and TCP frames from config/robots/<frames>.yaml."""
    robot = profile(name, tool)
    params = yaml.safe_load((_HSA_ROBOTS / f'{robot.frames}.yaml').read_text())['/**']['ros__parameters']
    assert params['base_frame'] == robot.base_frame
    assert params['tcp_frame'] == params['eef_frame'] == robot.tcp_frame


def test_every_laptop_robot_yaml_has_a_profile():
    """A yaml without a profile is a robot the laptop can select and the arm box cannot."""
    frames = {robots.get(name, tool).frames for name, tool in PROFILES}
    assert {p.stem for p in _HSA_ROBOTS.glob('*.yaml')} == frames


def test_fr3_without_a_tool_keeps_crbs_gripper():
    """The FR3 as CRB runs it: their fingertip frame, their payload, their hand frames, gripper drivers."""
    robot = robots.get('fr3', '')
    assert robot is robots.ROBOTS['fr3']
    assert (robot.calib, robot.tcp_frame, robot.frames) == ('tcp_calib', 'polyumi_tcp', 'fr3')
    assert robot.grippers


@pytest.mark.parametrize('tool', robots.HSA_TOOLS)
def test_an_hsa_tool_puts_the_fr3_on_hsa_calibration(tool):
    """An HSA policy is trained in hsa_tcp; on the FR3 it must see and command that frame, not polyumi_tcp."""
    robot = robots.get('fr3', tool)
    assert (robot.calib, robot.tcp_frame, robot.frames) == ('hsa_calib', 'hsa_tcp', 'fr3_hsa')
    assert (robot.flange_frames, robot.check_ee) == ('desk_ee', True)
    assert robot.servo_params == 'hsa_controllers_fr3.yaml'
    assert not robot.grippers
    # The arm itself is unchanged.
    crb = robots.ROBOTS['fr3']
    for field in ('arm_id', 'robot_ip', 'home_service', 'home_pose_file', 'arm_controller'):
        assert getattr(robot, field) == getattr(crb, field)


@pytest.mark.parametrize('tool', ['', *robots.HSA_TOOLS])
def test_the_fer_profile_does_not_depend_on_the_tool(tool):
    """The FER always carries an HSA tool; which one is hsa_calib's business, not the profile's."""
    assert robots.get('fer', tool) is robots.ROBOTS['fer']


def test_tool_defaults_to_the_environment(monkeypatch):
    """HSA_TOOL in the arm box's shell picks the tool, as station_env.sh exports it."""
    assert robots.get('fr3').calib == 'tcp_calib'
    monkeypatch.setenv(robots.TOOL_ENV, 'bare')
    assert robots.get('fr3').calib == 'hsa_calib'
    assert robots.get('fr3', '').calib == 'tcp_calib', 'an explicit empty tool beats the environment'


def test_unknown_tool_is_refused():
    """A typo must not fall back to CRB's gripper and its 0.7 kg payload."""
    with pytest.raises(ValueError, match='bare'):
        robots.get('fr3', 'bear')


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
