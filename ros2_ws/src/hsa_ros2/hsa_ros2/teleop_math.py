"""
The pure arithmetic of SpaceMouse teleoperation.

Numpy in, numpy out, so it is unit-testable in a plain venv (``pytest test/test_teleop_math.py``) without
ROS, a robot or a SpaceMouse; spacemouse_teleop.py reads the inputs and publishes the results.

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

    The SpaceMouse is a velocity device: full deflection moves the target by ``max_speed / hz`` per tick.
    ``rotation_mode`` (left button held) rotates the tool and locks translation; otherwise translation
    only. ``z_enabled`` (right button held) allows vertical motion; otherwise z is locked so an accidental
    push does not drive the tool into the table.

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


def extrapolate_chunk(target: np.ndarray, prev_target: np.ndarray | None, n: int, box: dict) -> list[np.ndarray]:
    """
    Build the ``n`` waypoints of a chunk: the target, then the target continued along its last step.

    Waypoint ``k`` is the target moved by ``k`` times the last tick's displacement (position linearly,
    orientation by composing the last tick's rotation ``k`` times), clipped to the box. A chunk that
    merely repeats the target hands the controller's monotone interpolant a zero tangent, so the
    reference would decelerate to a stop every tick; continuing along the velocity keeps it
    velocity-continuous, as a policy's chunk is. With no previous target (a fresh seed) or a zero step
    the chunk is a plain hold.
    """
    target = np.asarray(target, dtype=np.float64)
    if n <= 1 or prev_target is None:
        return [target.copy() for _ in range(max(1, n))]
    prev = np.asarray(prev_target, dtype=np.float64)
    delta_pos = target[:3] - prev[:3]
    rot_t = st.Rotation.from_rotvec(target[3:])
    delta_rot = rot_t * st.Rotation.from_rotvec(prev[3:]).inv()
    chunk = []
    for k in range(n):
        pose = np.empty(6)
        pose[:3] = target[:3] + k * delta_pos
        pose[3:] = ((delta_rot**k) * rot_t).as_rotvec() if k else target[3:]
        chunk.append(clamp_workspace(pose, box))
    return chunk


def pose6_to_quat(pose6: np.ndarray) -> np.ndarray:
    """Convert a rotation-vector pose to a quaternion ``[qx, qy, qz, qw]`` (the order geometry_msgs uses)."""
    return st.Rotation.from_rotvec(np.asarray(pose6)[3:]).as_quat()


def quat_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (geometry_msgs order) to a rotation vector."""
    return st.Rotation.from_quat([qx, qy, qz, qw]).as_rotvec()


def fake_sine_axes(t: float, period_s: float = 6.0, amplitude: float = 0.4) -> np.ndarray:
    """
    Stand in for the SpaceMouse with a slow circle in the base x-y plane, no rotation.

    ``input_source: fake_sine`` uses it to exercise the ROS plumbing on a desk with no device and, with
    ``execute: false``, no arm. Amplitude is a fraction of max speed.
    """
    w = 2.0 * np.pi / period_s
    return np.array([amplitude * np.cos(w * t), amplitude * np.sin(w * t), 0.0, 0.0, 0.0, 0.0])
