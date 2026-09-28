"""
The pure arithmetic of SpaceMouse teleoperation.

Everything here takes numpy arrays and returns numpy arrays, so it can be unit-tested in a
plain Python venv (``pytest test/test_teleop_math.py``) without ROS, a robot, or a SpaceMouse.
The node (spacemouse_teleop.py) is what read inputs, call these functions, and publishes the result.

Pose convention: a 6-vector ``[x, y, z, rx, ry, rz]`` in meters and a *rotation vector*
(axis-angle, scipy's ``rotvec``) expressed in the robot base frame.
"""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as st

# Maps SpaceMouse's axes onto robot base frame: robot = [-z, x, y] of device.
SPACEMOUSE_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def map_spacemouse_axes(raw: np.ndarray, deadzone: float = 0.0) -> np.ndarray:
    """
    Turn a raw SpaceMouse reading into a 6-vector of normalized velocities in the robot frame.

    :param raw: ``[x, y, z, roll, pitch, yaw]`` as pyspacemouse reports them, each in [-1, 1].
    :param deadzone: readings with |value| below this are zeroed, so a resting puck sends nothing.
    :returns: ``[vx, vy, vz, wx, wy, wz]`` in [-1, 1], robot base axes.
    """
    raw = np.asarray(raw, dtype=np.float64).copy()
    raw[np.abs(raw) < deadzone] = 0.0
    out = np.zeros(6)
    out[:3] = SPACEMOUSE_TO_ROBOT @ raw[:3]
    out[3:] = SPACEMOUSE_TO_ROBOT @ raw[3:]
    return out


def integrate_target(
    target: np.ndarray,
    axes: np.ndarray,
    *,
    max_pos_speed: float,
    max_rot_speed: float,
    hz: float,
    rotation_mode: bool,
    z_enabled: bool,
) -> np.ndarray:
    """
    Advance the commanded target pose by one control tick.

    The SpaceMouse is a *velocity* device: a deflection of 1.0 means "move at max speed". One tick
    therefore moves the target by ``max_speed / hz`` at full deflection. The target is what the
    controller is told to hold; the arm follows it through the impedance spring.

    Button semantics:

    * ``rotation_mode`` (left button held): the puck rotates the tool and translation is locked.
      Otherwise translation only, and rotation is locked.
    * ``z_enabled`` (right button held): vertical translation allowed. Otherwise z is locked so
      an accidental push does not drive the tool into the table.

    :param target: current ``[x, y, z, rx, ry, rz]`` target.
    :param axes: output of :func:`map_spacemouse_axes`.
    :returns: the new target (a new array; the input is not modified).
    """
    target = np.asarray(target, dtype=np.float64).copy()
    dpos = axes[:3] * (max_pos_speed / hz)
    drot = axes[3:] * (max_rot_speed / hz)
    if rotation_mode:
        dpos[:] = 0.0
    else:
        drot[:] = 0.0
    if not z_enabled:
        dpos[2] = 0.0
    target[:3] += dpos
    # Compose the small rotation ON THE LEFT: rotate about the base-frame
    # axes the puck names, rather than about the tool's own axes.
    delta = st.Rotation.from_euler('xyz', drot)
    target[3:] = (delta * st.Rotation.from_rotvec(target[3:])).as_rotvec()
    return target


def clamp_workspace(target: np.ndarray, box: dict) -> np.ndarray:
    """
    Clip the target position into an axis-aligned box in the base frame.

    :param box: ``{'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}`` in meters.
    """
    target = np.asarray(target, dtype=np.float64).copy()
    target[0] = np.clip(target[0], box['x_min'], box['x_max'])
    target[1] = np.clip(target[1], box['y_min'], box['y_max'])
    target[2] = np.clip(target[2], box['z_min'], box['z_max'])
    return target


def pose6_to_quat(pose6: np.ndarray) -> np.ndarray:
    """Rotation-vector pose -> quaternion ``[qx, qy, qz, qw]`` (the order geometry_msgs uses)."""
    return st.Rotation.from_rotvec(np.asarray(pose6)[3:]).as_quat()


def quat_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (geometry_msgs order) -> rotation vector."""
    return st.Rotation.from_quat([qx, qy, qz, qw]).as_rotvec()


def fake_sine_axes(t: float, period_s: float = 6.0, amplitude: float = 0.4) -> np.ndarray:
    """
    A stand-in for the SpaceMouse: a slow circle in the base x-y plane, no rotation.

    Used by ``input_source: fake_sine`` to test the whole ROS plumbing (TF lookup, chunk
    publishing, preview, recorder hooks) on a desk with no device attached and, with
    ``execute: false``, no arm. Amplitude is a fraction of max speed.
    """
    w = 2.0 * np.pi / period_s
    return np.array([amplitude * np.cos(w * t), amplitude * np.sin(w * t), 0.0, 0.0, 0.0, 0.0])
