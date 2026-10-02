"""
Unit tests for the ROS-free teleop arithmetic.

Run in any venv with numpy + scipy (no ROS needed):

    cd ~/hsa-ws/HSA-UMI/ros2_ws/src/hsa_ros2 && PYTHONPATH=. pytest test/test_teleop_math.py -q
"""

import numpy as np
import scipy.spatial.transform as st

from hsa_ros2.teleop_math import (
    extrapolate_chunk,
    clamp_workspace,
    fake_sine_axes,
    integrate_target,
    map_spacemouse_axes,
    pose6_to_quat,
    quat_to_rotvec,
)

BOX = dict(x_min=0.2, x_max=0.9, y_min=-0.4, y_max=0.4, z_min=0.05, z_max=0.5)


def test_frame_map_is_the_old_scripts_permutation():
    """Map the device axes to robot = [-z, x, y], for translation and rotation alike (tx_zup_spnav)."""
    axes = map_spacemouse_axes([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(axes[:3], [-0.3, 0.1, 0.2])
    assert np.allclose(axes[3:], [-0.6, 0.4, 0.5])


def test_deadzone_zeroes_small_readings():
    """Zero out readings below the deadzone."""
    axes = map_spacemouse_axes([0.05, -0.05, 0.05, 0.0, 0.0, 0.0], deadzone=0.1)
    assert np.allclose(axes, 0.0)


def test_translation_only_without_rotation_button():
    """Translate only, with rotation locked, while the rotation button is not held."""
    target = np.array([0.5, 0.0, 0.3, np.pi, 0.0, 0.0])
    axes = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # full x push AND full roll
    new = integrate_target(
        target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0, rotation_mode=False, z_enabled=False
    )
    assert np.isclose(new[0] - target[0], 0.02)  # 0.2 m/s / 10 Hz
    assert np.allclose(new[3:], target[3:])  # rotation locked


def test_rotation_mode_locks_translation():
    """Rotate about the base axes only, with translation locked, while the rotation button is held."""
    target = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    axes = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # push AND yaw
    new = integrate_target(
        target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0, rotation_mode=True, z_enabled=True
    )
    assert np.allclose(new[:3], target[:3])  # translation locked
    yaw = st.Rotation.from_rotvec(new[3:]).as_euler('xyz')[2]
    assert np.isclose(yaw, 0.015)  # 0.15 rad/s / 10 Hz about base z


def test_z_locked_unless_enabled():
    """Lock z unless the z button is held."""
    target = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    axes = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    locked = integrate_target(
        target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0, rotation_mode=False, z_enabled=False
    )
    free = integrate_target(
        target, axes, max_pos_speed=0.2, max_rot_speed=0.15, hz=10.0, rotation_mode=False, z_enabled=True
    )
    assert np.isclose(locked[2], 0.3) and np.isclose(free[2], 0.32)


def test_clamp_keeps_pose_inside_box():
    """Clip the position into the box."""
    out = clamp_workspace(np.array([1.5, -0.9, -0.1, 0.0, 0.0, 0.0]), BOX)
    assert np.allclose(out[:3], [0.9, -0.4, 0.05])


def test_quaternion_round_trip():
    """Convert a rotation vector to a quaternion and back."""
    pose = np.array([0.0, 0.0, 0.0, 0.3, -0.2, 1.1])
    q = pose6_to_quat(pose)
    assert np.allclose(quat_to_rotvec(*q), pose[3:])


def test_fake_sine_is_bounded_and_planar():
    """Keep the fake sine within the amplitude and in the x-y plane."""
    for t in np.linspace(0, 12, 50):
        a = fake_sine_axes(t)
        assert np.all(np.abs(a) <= 0.4 + 1e-9) and np.allclose(a[2:], 0.0)


CHUNK_BOX = {'x_min': 0.2, 'x_max': 0.7, 'y_min': -0.4, 'y_max': 0.4, 'z_min': 0.05, 'z_max': 0.5}


def test_chunk_continues_along_the_last_step():
    """Waypoint k sits k steps past the target, so the reference keeps the puck's velocity between ticks."""
    prev = np.array([0.49, 0.0, 0.3, 0.0, 0.0, 0.0])
    target = np.array([0.50, 0.0, 0.3, 0.0, 0.0, 0.0])

    chunk = extrapolate_chunk(target, prev, 3, CHUNK_BOX)

    assert [round(p[0], 6) for p in chunk] == [0.50, 0.51, 0.52]
    assert all(np.allclose(p[1:], target[1:]) for p in chunk)


def test_chunk_is_a_hold_without_a_previous_target_or_without_motion():
    """A fresh seed, a snapped target, or a still puck must not invent a velocity."""
    target = np.array([0.5, 0.1, 0.3, 0.0, 0.0, 0.0])

    assert all(np.array_equal(p, target) for p in extrapolate_chunk(target, None, 3, CHUNK_BOX))
    assert all(np.array_equal(p, target) for p in extrapolate_chunk(target, target, 3, CHUNK_BOX))
    assert len(extrapolate_chunk(target, None, 1, CHUNK_BOX)) == 1


def test_chunk_composes_the_last_rotation_step():
    """A 0.1 rad yaw step per tick extrapolates to 0.2 and 0.3 rad, about the same axis."""
    prev = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    target = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.1])

    chunk = extrapolate_chunk(target, prev, 3, CHUNK_BOX)

    assert [round(p[5], 6) for p in chunk] == [0.1, 0.2, 0.3]


def test_chunk_stays_inside_the_workspace_box():
    """Extrapolation past the box edge is clipped like the target itself."""
    prev = np.array([0.68, 0.0, 0.3, 0.0, 0.0, 0.0])
    target = np.array([0.69, 0.0, 0.3, 0.0, 0.0, 0.0])

    chunk = extrapolate_chunk(target, prev, 4, CHUNK_BOX)

    assert [round(p[0], 6) for p in chunk] == [0.69, 0.70, 0.70, 0.70]
