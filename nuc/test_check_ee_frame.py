r"""
Tests for the bringup-time end-effector frame check.

The two candidate frames differ by 10 cm and 45 deg, so the comparison is undemanding; what matters is
that each outcome maps to the right exit code, because bringup.launch.py aborts on 1 and continues on
0. Runs on the laptop: only franka_msgs' message definitions are needed, and the one test that spins a
node does so against an empty graph.

    bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \\
      && source ~/ws/franka/install/setup.bash \\
      && /usr/bin/python3 -m pytest nuc/test_check_ee_frame.py -q'
"""

import math

import pytest

import check_ee_frame
import hsa_calib

IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


def test_quaternion_from_rpy_matches_tf2_for_a_pure_yaw():
    """Rz(-45 deg) is (0, 0, -sin(22.5), cos(22.5)) in tf2's (x, y, z, w) order."""
    q = check_ee_frame.quaternion_from_rpy(0.0, 0.0, -math.pi / 4)

    assert q == pytest.approx((0.0, 0.0, -math.sin(math.pi / 8), math.cos(math.pi / 8)))


def test_identity_report_matches_desk_none():
    """A robot with no end effector reports F_T_EE = identity, which is the `none` transform."""
    xyz, rpy = hsa_calib.ee_transform('none')
    ok, detail = check_ee_frame.compare((0.0, 0.0, 0.0), IDENTITY_Q, xyz, rpy)

    assert ok, detail


def test_hand_report_does_not_match_desk_none():
    """A robot configured with the Franka Hand must be caught when the launch assumed `none`."""
    hand_xyz, hand_rpy = hsa_calib.ee_transform('hand')
    hand_q = check_ee_frame.quaternion_from_rpy(*hand_rpy)
    none_xyz, none_rpy = hsa_calib.ee_transform('none')

    ok, detail = check_ee_frame.compare(hand_xyz, hand_q, none_xyz, none_rpy)

    assert not ok
    assert '103.4 mm' in detail


def test_hand_report_matches_desk_hand_within_float_noise():
    """The same pose round-tripped through a quaternion is still a match — the tolerances are loose enough."""
    xyz, rpy = hsa_calib.ee_transform('hand')
    q = check_ee_frame.quaternion_from_rpy(*rpy)
    noisy = (xyz[0] + 1e-5, xyz[1] - 1e-5, xyz[2] + 1e-5)

    ok, _ = check_ee_frame.compare(noisy, q, xyz, rpy)

    assert ok


def test_a_flipped_quaternion_is_the_same_rotation():
    """Q and -q describe one rotation; libfranka's conversion may hand back either sign."""
    xyz, rpy = hsa_calib.ee_transform('hand')
    q = tuple(-v for v in check_ee_frame.quaternion_from_rpy(*rpy))

    ok, _ = check_ee_frame.compare(xyz, q, xyz, rpy)

    assert ok


def test_nothing_published_is_a_failure_not_a_pass(monkeypatch):
    """
    No robot_state within the timeout exits 1.

    "Cannot verify" is reserved for a message without the field; silence means the broadcaster is
    not up, and bringing the arm up past a dead broadcaster hides every later check too.

    On a private domain with discovery off: on the lab laptop this test otherwise hears the real
    station's broadcaster over the cable and passes the check it is asserting fails.
    """
    monkeypatch.setenv('ROS_DOMAIN_ID', '97')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'OFF')
    assert check_ee_frame.main(['0.3', '--calib', 'hsa_calib']) == 1


def test_unknown_desk_setting_fails_before_touching_ros():
    """A typo on the launch line must surface as the ValueError hsa_calib raises, not as a mismatch."""
    with pytest.raises(ValueError):
        check_ee_frame.main(['0.3', '--calib', 'hsa_calib', '--desk-end-effector', 'gripper'])
