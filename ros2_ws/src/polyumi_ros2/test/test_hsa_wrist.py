"""The HSA wrist observation contract shared by the recorder and the policy client (no ROS)."""

import numpy as np
import pytest

from polyumi_ros2 import hsa_wrist as hw


class _FakeWristState:
    extension_deg = [360.0] * 6
    command_deg = [360.0] * 6
    motor_pos_turns = [1.0, 2.0]  # short on purpose
    motor_vel_turns_s = [0.0] * 6
    torque_estimate_nm = [0.1] * 6
    torque_target_nm = [0.0] * 6
    iq_setpoint_a = [0.0] * 6
    iq_measured_a = [2.0] * 6
    bus_voltage_v = [24.0] * 6
    bus_current_a = [0.5] * 6
    fet_temperature_c = [30.0] * 6
    motor_temperature_c = [35.0] * 6
    controller_errors = [0, 0, 0, 4, 0, 0]
    axis_state = [8] * 7  # long on purpose


def test_wrist_state_arrays_give_six_columns_per_key():
    arrays = hw.wrist_state_arrays(_FakeWristState())
    assert set(arrays) == set(hw.WRIST_STATE_FIELDS) and all(a.shape == (6,) for a in arrays.values())
    assert arrays[hw.KEY_WRIST_MOTOR_POS][1] == 2.0 and np.isnan(arrays[hw.KEY_WRIST_MOTOR_POS][2])
    assert arrays[hw.KEY_WRIST_ERRORS][3] == 4 and arrays[hw.KEY_WRIST_AXIS_STATE].tolist() == [8] * 6


def test_plate_arrays_hold_the_pose_and_flag_freshness():
    fresh = hw.plate_arrays(np.array([0.0, 0.0, -0.012, 0.0, 0.01, 0.0]), True)
    assert fresh[hw.KEY_PLATE_POS].tolist() == [0.0, 0.0, -0.012] and fresh[hw.KEY_PLATE_VISIBLE].tolist() == [1.0]
    held = hw.plate_arrays(np.array([0.0, 0.0, -0.012, 0.0, 0.01, 0.0]), False)
    assert held[hw.KEY_PLATE_ROT].tolist() == [0.0, 0.01, 0.0] and held[hw.KEY_PLATE_VISIBLE].tolist() == [0.0]
    assert np.isnan(hw.plate_arrays(None, False)[hw.KEY_PLATE_POS]).all()


def test_resize_camera1_keeps_the_whole_frame():
    pytest.importorskip('cv2')
    img = np.zeros((800, 1280, 3), dtype=np.uint8)
    img[:, :40] = 255
    img[:, -40:] = 255
    out = hw.resize_camera1_rgb(img)
    assert out.shape == (224, 224, 3) and out[100, 0, 0] > 200 and out[100, -1, 0] > 200 and out[100, 112, 0] == 0
