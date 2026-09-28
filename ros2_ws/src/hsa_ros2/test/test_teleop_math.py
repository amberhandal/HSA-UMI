"""
Unit tests for the ROS-free teleop arithmetic.

Run in any venv with numpy + scipy (no ROS needed):

    cd ~/hsa-ws/HSA-UMI/ros2_ws/src/hsa_ros2 && PYTHONPATH=. pytest test/test_teleop_math.py -q
"""

import numpy as np
import scipy.spatial.transform as st

from hsa_ros2.teleop_math import (
    clamp_workspace,
    fake_sine_axes,
    integrate_target,
    map_spacemouse_axes,
    pose6_to_quat,
    quat_to_rotvec,
)

BOX = dict(x_min=0.2, x_max=0.9, y_min=-0.4, y_max=0.4, z_min=0.05, z_max=0.5)


def test_frame_map_is_the_old_scripts_permutation():
    # robot = [-z, x, y] of the device, for translation and rotation alike (tx_zup_spnav).
    axes = map_spacemouse_axes([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(axes[:3], [-0.3, 0.1, 0.2])
    assert np.allclose(axes[3:], [-0.6, 0.4, 0.5])


def test_deadzone_zeroes_small_readings():
    axes = map_spacemouse_axes([0.05, -0.05, 0.05, 0.0, 0.0, 0.0], deadzone=0.1)
    assert np.allclose(axes, 0.0)


def test_translation_only_without_rotation_button():
    target = np.array([0.5, 0.0, 0.3, np.pi, 0.0, 0.0])
    axes = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # full x push AND full roll
    new = integrate_target(target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0,
                           rotation_mode=False, z_enabled=False)
    assert np.isclose(new[0] - target[0], 0.02)          # 0.2 m/s / 10 Hz
    assert np.allclose(new[3:], target[3:])              # rotation locked


def test_rotation_mode_locks_translation():
    target = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    axes = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])      # push AND yaw
    new = integrate_target(target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0,
                           rotation_mode=True, z_enabled=True)
    assert np.allclose(new[:3], target[:3])              # translation locked
    yaw = st.Rotation.from_rotvec(new[3:]).as_euler('xyz')[2]
    assert np.isclose(yaw, 0.015)                        # 0.15 rad/s / 10 Hz about base z


def test_z_locked_unless_enabled():
    target = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    axes = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    locked = integrate_target(target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0,
                              rotation_mode=False, z_enabled=False)
    free = integrate_target(target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0,
                            rotation_mode=False, z_enabled=True)
    assert np.isclose(locked[2], 0.3) and np.isclose(free[2], 0.32)


def test_clamp_keeps_pose_inside_box():
    out = clamp_workspace(np.array([1.5, -0.9, -0.1, 0.0, 0.0, 0.0]), BOX)
    assert np.allclose(out[:3], [0.9, -0.4, 0.05])


def test_quaternion_round_trip():
    pose = np.array([0.0, 0.0, 0.0, 0.3, -0.2, 1.1])
    q = pose6_to_quat(pose)
    assert np.allclose(quat_to_rotvec(*q), pose[3:])


def test_fake_sine_is_bounded_and_planar():
    for t in np.linspace(0, 12, 50):
        a = fake_sine_axes(t)
        assert np.all(np.abs(a) <= 0.4 + 1e-9) and np.allclose(a[2:], 0.0)
