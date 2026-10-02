r"""
Tests for the station-side /polyumi/home and /polyumi/set_home services on the FER.

Homing is the one path here that moves the arm on an explicit request rather than a streamed chunk, and
every way it can be wrong is silent: a wrong joint name drives a different joint, a stale delta runs the
sweep at the wrong speed, and a failed hand-back leaves the policy commanding a controller that no longer
holds the arm. Runs on the laptop: the action and service clients are mocked, so no station is involved.

    bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \\
      && /usr/bin/python3 -m pytest nuc/test_fer_home_service.py -q'
"""

import math
from unittest.mock import MagicMock, patch

from control_msgs.action import FollowJointTrajectory
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
import yaml

import fer_home_service as hs


@pytest.fixture(scope='module', autouse=True)
def ros():
    """Init rclpy once for the module; every node here is constructed without a real executor."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node(tmp_path):
    """
    Build a service node whose action and switch clients are mocks, so no server is required.

    `home_pose_file` defaults into tmp_path: the node loads a taught pose at construction, and
    without this every test on a machine that has ever run /polyumi/set_home would home to
    whatever is in the developer's ~/.ros.
    """
    nodes = []

    def _make(**overrides):
        overrides.setdefault('home_pose_file', str(tmp_path / 'home_pose.yaml'))
        params = [Parameter(k, value=v) for k, v in overrides.items()]
        with (
            patch.object(hs.FerHomeService, 'create_client', side_effect=lambda *a, **k: MagicMock()),
            patch.object(hs, 'ActionClient') as action_client,
        ):
            action_client.return_value.wait_for_server.return_value = True
            action_client.return_value.server_is_ready.return_value = True
            node = hs.FerHomeService(parameter_overrides=params)
        node.get_logger = MagicMock()
        # Futures never resolve against a mocked client, so _wait would burn its full timeout.
        node._wait = lambda future, timeout_s: True
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()


def _feed_joint_state(node, positions=None):
    """Deliver one /joint_states message, as the subscription callback would."""
    msg = JointState()
    msg.name = list(hs.HOME_JOINT_NAMES) + ['some_other_joint']
    msg.position = list(positions if positions is not None else [0.1] * 7) + [0.0]
    node._on_joint_state(msg)


def _capture_goal(node, ok=True, why=''):
    """Replace _run_goal with a recorder; returns the list it appends (trajectory, timeout) to."""
    calls = []

    def _record(trajectory, timeout_s):
        calls.append((trajectory, timeout_s))
        return ok, why

    node._run_goal = _record
    return calls


def _home(node) -> Trigger.Response:
    return node._on_home(Trigger.Request(), Trigger.Response())


def _set_home(node) -> Trigger.Response:
    return node._on_set_home(Trigger.Request(), Trigger.Response())


# ----------------------------------------------------------------------
# The move
# ----------------------------------------------------------------------


def test_duration_is_the_largest_delta_at_max_speed_never_under_the_minimum():
    """One joint 0.6 rad away at 0.3 rad/s takes 2 s, but never less than the 3 s floor."""
    assert hs.home_duration_s([0.0] * 7, [0.6] + [0.0] * 6, 0.3, 3.0) == pytest.approx(3.0)
    assert hs.home_duration_s([0.0] * 7, [0.0] * 6 + [3.0], 0.3, 3.0) == pytest.approx(10.0)


def test_home_sends_one_point_at_the_ready_pose_at_rest(make_node):
    """The goal names fer_joint1..7 at franka's `ready` values, zero end velocity, timed from the delta."""
    node = make_node()
    _feed_joint_state(node, [0.0] * 7)
    goals = _capture_goal(node)

    assert _home(node).success

    trajectory, timeout_s = goals[0]
    assert trajectory.joint_names == [f'fer_joint{i}' for i in range(1, 8)]
    point = trajectory.points[0]
    assert list(point.positions) == pytest.approx(hs.HOME_JOINTS)
    assert list(point.velocities) == [0.0] * 7
    expected = hs.home_duration_s([0.0] * 7, hs.HOME_JOINTS, hs.DEFAULT_MAX_JOINT_SPEED, hs.DEFAULT_MIN_DURATION_S)
    assert point.time_from_start.sec + point.time_from_start.nanosec / 1e9 == pytest.approx(expected, abs=1e-6)
    assert timeout_s == pytest.approx(expected + hs.HOME_EXECUTE_MARGIN_S)


def test_home_refused_without_a_fresh_joint_state(make_node):
    """The duration comes from the delta and the delta from /joint_states; no state, no move."""
    node = make_node()
    goals = _capture_goal(node)

    response = _home(node)

    assert not response.success
    assert 'cannot read the arm pose' in response.message
    assert goals == []


def test_home_refused_while_a_move_is_in_flight(make_node):
    """A concurrent /polyumi/home must not cut in on one already executing."""
    node = make_node()
    _feed_joint_state(node)
    goals = _capture_goal(node)
    node._busy.acquire()
    try:
        response = _home(node)
    finally:
        node._busy.release()

    assert not response.success
    assert 'busy' in response.message
    assert goals == []


def test_home_releases_the_busy_lock_after_a_failed_move(make_node):
    """A failed goal must not wedge the service."""
    node = make_node()
    _feed_joint_state(node)
    _capture_goal(node, ok=False, why='goal rejected')

    assert not _home(node).success
    assert node._busy.acquire(blocking=False), 'busy lock left held after a failed home'
    node._busy.release()


def test_home_rejects_a_wrong_length_home_joints(make_node):
    """A 6-value override would otherwise zip() short and home to a partial pose."""
    node = make_node(home_joints=[0.0] * 6)
    _feed_joint_state(node)
    goals = _capture_goal(node)

    response = _home(node)

    assert not response.success
    assert 'expected 7' in response.message
    assert goals == []


def test_goal_timeout_cancels_and_waits_for_the_abort(make_node):
    """A timed-out goal keeps driving the arm; the cancel must land before the busy lock is released."""
    node = make_node()
    waits = []

    def _wait(_future, timeout_s):
        waits.append(timeout_s)
        return len(waits) != 2  # only the goal-result wait times out

    node._wait = _wait

    ok, why = node._run_goal(hs.JointTrajectory(), timeout_s=7.0)

    assert not ok and 'timed out' in why
    goal_handle = node._exec.send_goal_async.return_value.result.return_value
    goal_handle.cancel_goal_async.assert_called_once()
    assert waits == [hs.GOAL_SUBMIT_TIMEOUT_S, 7.0, hs.CANCEL_TIMEOUT_S]


def test_goal_success_does_not_cancel(make_node):
    """The cancel path must not fire on the happy path."""
    node = make_node()
    handle = node._exec.send_goal_async.return_value.result.return_value
    result = handle.get_result_async.return_value.result.return_value
    result.result.error_code = FollowJointTrajectory.Result.SUCCESSFUL

    ok, _ = node._run_goal(hs.JointTrajectory(), timeout_s=7.0)

    assert ok
    handle.cancel_goal_async.assert_not_called()


def test_rejected_goal_names_the_controller(make_node):
    """A rejected goal usually means fer_arm_controller is loaded but inactive; say so."""
    node = make_node()
    node._exec.send_goal_async.return_value.result.return_value.accepted = False

    ok, why = node._run_goal(hs.JointTrajectory(), timeout_s=7.0)

    assert not ok and 'fer_arm_controller' in why


# ----------------------------------------------------------------------
# Controller handover
# ----------------------------------------------------------------------


def _stub_switch_sequence(node, oks):
    """Each successive switch_controller call answers the next `oks` value; returns call_async."""
    node._switch.wait_for_service.return_value = True
    futures = []
    for ok in oks:
        result = MagicMock()
        result.ok = ok
        future = MagicMock()
        future.result.return_value = result
        futures.append(future)
    node._switch.call_async.side_effect = futures
    return node._switch.call_async


def _switch_pairs(call_async):
    return [
        (tuple(c[0][0].activate_controllers), tuple(c[0][0].deactivate_controllers)) for c in call_async.call_args_list
    ]


def test_home_borrows_the_arm_from_the_servo_and_gives_it_back(make_node):
    """Both controllers claim the same effort interfaces; homing must swap and swap back."""
    node = make_node()
    _feed_joint_state(node)
    _capture_goal(node)
    call_async = _stub_switch_sequence(node, [True, True])

    assert _home(node).success

    assert _switch_pairs(call_async) == [
        ((hs.TRAJECTORY_CONTROLLER,), (hs.SERVO_CONTROLLER,)),
        ((hs.SERVO_CONTROLLER,), (hs.TRAJECTORY_CONTROLLER,)),
    ]
    assert call_async.call_args_list[0][0][0].strictness == hs.SwitchController.Request.STRICT
    assert all(c[0][0].activate_asap for c in call_async.call_args_list), 'switch inside the realtime loop'


def test_home_reports_failure_if_the_servo_cannot_be_reactivated(make_node):
    """`homed: true` with the arm left on fer_arm_controller would be a silently dead policy."""
    node = make_node()
    _feed_joint_state(node)
    _capture_goal(node)
    _stub_switch_sequence(node, [True, False])

    response = _home(node)

    assert not response.success
    assert 'failed to hand the arm back' in response.message


def test_home_without_the_servo_running_does_not_switch_back(make_node):
    """A declined borrow (servo never spawned) must not be followed by activating it."""
    node = make_node()
    _feed_joint_state(node)
    _capture_goal(node)
    call_async = _stub_switch_sequence(node, [False])

    assert _home(node).success
    assert len(call_async.call_args_list) == 1


# ----------------------------------------------------------------------
# Teaching the start pose
# ----------------------------------------------------------------------


def test_set_home_records_the_current_pose_and_home_replays_it(make_node):
    """The taught pose replaces the default for the next /polyumi/home."""
    node = make_node()
    taught = [0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 0.7]
    _feed_joint_state(node, taught)
    goals = _capture_goal(node)

    assert _set_home(node).success
    assert _home(node).success

    assert list(goals[0][0].points[0].positions) == pytest.approx(taught)


def test_set_home_persists_and_a_new_node_loads_it(make_node, tmp_path):
    """A pose taught before a bringup restart must survive it, and say where it came from."""
    path = tmp_path / 'persisted.yaml'
    node = make_node(home_pose_file=str(path))
    taught = [0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 0.7]
    _feed_joint_state(node, taught)

    assert _set_home(node).success
    doc = yaml.safe_load(path.read_text())
    assert doc['joint_names'] == hs.HOME_JOINT_NAMES

    fresh = make_node(home_pose_file=str(path))
    assert fresh._home_joints == pytest.approx(taught)
    assert 'taught pose' in fresh._home_source


def test_a_pose_file_from_another_arm_is_ignored(make_node, tmp_path):
    """fr3_joint names replayed onto fer_joint1..7 by position would home somewhere else entirely."""
    path = tmp_path / 'fr3.yaml'
    path.write_text(
        yaml.safe_dump(
            {'version': 1, 'joint_names': [f'fr3_joint{i}' for i in range(1, 8)], 'joint_positions': [0.5] * 7}
        )
    )

    node = make_node(home_pose_file=str(path))

    assert node._home_joints == pytest.approx(hs.HOME_JOINTS)


def test_set_home_refused_without_a_fresh_joint_state(make_node):
    """Recording a stale cache would teach wherever the arm used to be."""
    node = make_node()

    response = _set_home(node)

    assert not response.success
    assert 'cannot read the arm pose' in response.message


def test_set_home_refused_mid_home(make_node):
    """The arm is merely passing through while a home is in flight."""
    node = make_node()
    _feed_joint_state(node)
    node._busy.acquire()
    try:
        response = _set_home(node)
    finally:
        node._busy.release()

    assert not response.success
    assert 'busy' in response.message


def test_ready_pose_is_frankas():
    """The default home is franka's own `ready` configuration."""
    assert hs.HOME_JOINTS == pytest.approx([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])


def test_home_pose_file_none_means_the_ready_pose_and_no_persistence(make_node):
    """`home_pose_file:=none` (a shell strips empty quotes) homes to franka's ready pose and saves nothing."""
    node = make_node(home_pose_file='none')
    _feed_joint_state(node, [0.1] * 7)

    assert node._home_joints == pytest.approx(hs.HOME_JOINTS)
    response = _set_home(node)
    assert response.success and 'not saved' in response.message
