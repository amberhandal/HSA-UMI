"""
Guard what hsa_calib computes rather than states, and the one frame that depends on a Desk setting.

Both fail invisibly on the robot: a zero inertia tensor is accepted by every layer of ROS and
rejected only by the arm itself, and a fer_hand_tcp that does not match Desk's end effector moves
the impedance spring's anchor by 10 cm with nothing on the wire to say so.
"""

import math

import pytest

import hsa_calib


def test_shipped_payload_constants_give_an_inertia_the_arm_accepts():
    """The FER rejects mass>0 carrying a zero tensor as a bare "invalid argument"."""
    assert hsa_calib.PAYLOAD_MASS > 0
    assert all(v > 0 for v in hsa_calib.payload_inertia_flange()[::4])  # the three diagonal terms


def test_inertia_is_diagonal_in_the_flange():
    """The box extents are stated in fer_link8 itself, so no rotation may introduce cross terms."""
    _, fxy, fxz, fyx, _, fyz, fzx, fzy, _ = hsa_calib.payload_inertia_flange()

    assert (fxy, fxz, fyx, fyz, fzx, fzy) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_com_is_passed_through_unchanged():
    """set_payload reads payload_com_flange(); it must be the flange-frame constant, not a copy."""
    assert hsa_calib.payload_com_flange() == hsa_calib.PAYLOAD_COM_FLANGE


def test_tcp_hangs_off_the_flange_not_a_hand():
    """There is no hand on this arm; hsa_tcp's parent must be fer_link8 so TF has a path to it."""
    args = hsa_calib.static_transform_publisher_args()

    assert args[args.index('--frame-id') + 1] == 'fer_link8'
    assert args[args.index('--child-frame-id') + 1] == 'hsa_tcp'
    assert float(args[args.index('--z') + 1]) == pytest.approx(hsa_calib.PLATE_HEIGHT_Z)


def test_ee_frame_for_desk_none_is_the_flange_itself():
    """With no end effector in Desk, F_T_EE is identity: fer_hand_tcp coincides with fer_link8."""
    xyz, rpy = hsa_calib.ee_transform('none')

    assert xyz == (0.0, 0.0, 0.0)
    assert rpy == (0.0, 0.0, 0.0)


def test_ee_frame_for_desk_hand_is_franka_descriptions_hand_tcp():
    """
    The Franka Hand setting composes fer_link8 -> fer_hand (Rz -45 deg) with fer_hand -> fer_hand_tcp (z 0.1034).

    A yaw leaves a pure z offset untouched, so the composed transform is that offset with that yaw.
    """
    xyz, rpy = hsa_calib.ee_transform('hand')

    assert xyz == pytest.approx((0.0, 0.0, 0.1034))
    assert rpy == pytest.approx((0.0, 0.0, -math.pi / 4))


def test_ee_frame_publisher_names_frankas_ee_frame():
    """The controller looks up `<arm_id>_hand_tcp -> tcp_frame`, so the child must be fer_hand_tcp."""
    args = hsa_calib.ee_transform_publisher_args('none')

    assert args[args.index('--frame-id') + 1] == 'fer_link8'
    assert args[args.index('--child-frame-id') + 1] == 'fer_hand_tcp'


def test_unknown_desk_setting_is_refused_not_guessed():
    """A typo must not silently become identity — that is the exact mistake the check exists for."""
    with pytest.raises(ValueError, match='unknown Desk end effector'):
        hsa_calib.ee_transform('gripper')


def test_default_desk_setting_is_a_known_one():
    """The launch argument's default and choices both come from these two names."""
    assert hsa_calib.DESK_END_EFFECTOR in hsa_calib.DESK_EE_TRANSFORMS


def test_bringup_log_lines_name_the_calibration_status():
    """Until the ESTIMATE placeholders are measured, every bringup must say so where it is read."""
    assert hsa_calib.CALIBRATION_STATUS in hsa_calib.describe()
    assert hsa_calib.CALIBRATION_STATUS in hsa_calib.describe_payload()


def test_bare_flange_preset_puts_the_tcp_on_the_flange_with_no_payload(monkeypatch):
    """HSA_TOOL=bare is the first-session configuration: nothing bolted on, nothing to compensate."""
    import importlib

    monkeypatch.setenv('HSA_TOOL', 'bare')
    bare = importlib.reload(hsa_calib)
    try:
        assert bare.TCP_XYZ == (0.0, 0.0, 0.0)
        assert bare.PAYLOAD_MASS == 0.0
        assert all(v == 0.0 for v in bare.payload_inertia_flange())
        assert 'bare' in bare.describe()
    finally:
        monkeypatch.delenv('HSA_TOOL')
        importlib.reload(hsa_calib)
