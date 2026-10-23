"""
The HSA wrist's observation contract, shared by the recorder and the policy client.

Both ends must turn the same messages into the same arrays: ``hsa_wrist_msgs/WristState`` into the
per-axis ``robot0_wrist_*`` columns, ``/platform_pose`` into the plate's displacement, and the
baseplate camera frame into ``camera1_rgb``. Anything that differs between the two is train/serve
skew the policy cannot report, so the field table and the resize live here once, like
``camera_preproc`` does for ``camera0_rgb``. Pure numpy; the message type is never imported, only
read by attribute, so this module loads without the wrist workspace.
"""

from __future__ import annotations

import numpy as np

#: The policy grid; the same 224 as camera0_rgb.
CAMERA1_RGB_RESOLUTION = 224

# Dataset keys, one column per HSA / ODrive axis (index = node id), from /hsa/wrist/state.
KEY_WRIST_EXT = 'robot0_wrist_ext_deg'  # measured extension angle (deg)
KEY_WRIST_CMD = 'robot0_wrist_cmd_deg'  # commanded extension angle (deg)
KEY_WRIST_MOTOR_POS = 'robot0_wrist_motor_pos'  # ODrive pos_estimate (turns)
KEY_WRIST_MOTOR_VEL = 'robot0_wrist_motor_vel'  # ODrive vel_estimate (turns/s)
KEY_WRIST_TORQUE = 'robot0_wrist_torque'  # torque_estimate (Nm)
KEY_WRIST_TORQUE_TARGET = 'robot0_wrist_torque_target'  # torque_target (Nm)
KEY_WRIST_IQ = 'robot0_wrist_iq'  # iq_measured (A), the motor current
KEY_WRIST_IQ_SETPOINT = 'robot0_wrist_iq_setpoint'  # iq_setpoint (A)
KEY_WRIST_BUS_V = 'robot0_wrist_bus_voltage'  # DC bus voltage per ODrive (V)
KEY_WRIST_BUS_I = 'robot0_wrist_bus_current'  # DC bus current per ODrive (A)
KEY_WRIST_FET_TEMP = 'robot0_wrist_fet_temp'  # ODrive FET temperature (C)
KEY_WRIST_MOTOR_TEMP = 'robot0_wrist_motor_temp'  # motor temperature (C)
KEY_WRIST_ERRORS = 'robot0_wrist_errors'  # ODrive active_errors bitmask
KEY_WRIST_AXIS_STATE = 'robot0_wrist_axis_state'  # ODrive axis state (8 = closed loop)
# The top plate seen by the baseplate camera: /platform_pose, the plate in its rest frame
# platform_zero, so it is the plate's displacement from rest. z points out of the tag face toward
# the camera, so extension reads negative and compression under a push positive.
KEY_PLATE_POS = 'robot0_plate_pos'  # m
KEY_PLATE_ROT = 'robot0_plate_rot_axis_angle'  # rotation vector (rad)
KEY_PLATE_VISIBLE = 'robot0_plate_visible'  # 1 when the pose is fresh (tags seen), else the pose is held
KEY_WRIST_IMAGE = 'camera1_rgb'  # the baseplate camera frame, resized

#: Axes per wrist.
N_AXES = 6

#: The WristState message field behind each wrist key, in the order the keys are defined.
WRIST_STATE_FIELDS = {
    KEY_WRIST_EXT: 'extension_deg',
    KEY_WRIST_CMD: 'command_deg',
    KEY_WRIST_MOTOR_POS: 'motor_pos_turns',
    KEY_WRIST_MOTOR_VEL: 'motor_vel_turns_s',
    KEY_WRIST_TORQUE: 'torque_estimate_nm',
    KEY_WRIST_TORQUE_TARGET: 'torque_target_nm',
    KEY_WRIST_IQ: 'iq_measured_a',
    KEY_WRIST_IQ_SETPOINT: 'iq_setpoint_a',
    KEY_WRIST_BUS_V: 'bus_voltage_v',
    KEY_WRIST_BUS_I: 'bus_current_a',
    KEY_WRIST_FET_TEMP: 'fet_temperature_c',
    KEY_WRIST_MOTOR_TEMP: 'motor_temperature_c',
    KEY_WRIST_ERRORS: 'controller_errors',
    KEY_WRIST_AXIS_STATE: 'axis_state',
}
#: The two integer columns; everything else is a float.
WRIST_INT_KEYS = (KEY_WRIST_ERRORS, KEY_WRIST_AXIS_STATE)
#: The plate keys with their widths.
PLATE_KEYS = {KEY_PLATE_POS: 3, KEY_PLATE_ROT: 3, KEY_PLATE_VISIBLE: 1}


def wrist_state_arrays(msg, n_axes: int = N_AXES) -> dict[str, np.ndarray]:
    """
    Pull the per-axis columns out of a ``hsa_wrist_msgs/WristState`` (or any object with its fields).

    Each array is float64 of length ``n_axes``, padded with NaN (ints with 0) so a short message
    cannot shift columns. The dataset writer casts to its own dtypes; the policy client sends float32.
    """
    out = {}
    for key, field in WRIST_STATE_FIELDS.items():
        vals = list(getattr(msg, field))[:n_axes]
        fill = 0 if key in WRIST_INT_KEYS else np.nan
        out[key] = np.array(vals + [fill] * (n_axes - len(vals)), dtype=np.float64)
    return out


def plate_arrays(pose6: np.ndarray | None, fresh: bool) -> dict[str, np.ndarray]:
    """
    The plate's three keys from a ``[pos, rotvec]`` pose in platform_zero (NaN when none yet).

    ``fresh`` is the recorder's rule: the newest pose is no older than ``plate.max_age_s`` and the
    tags were seen; otherwise the pose is held and ``robot0_plate_visible`` is 0.
    """
    p = np.full(6, np.nan) if pose6 is None else np.asarray(pose6, dtype=np.float64)
    return {
        KEY_PLATE_POS: p[:3],
        KEY_PLATE_ROT: p[3:],
        KEY_PLATE_VISIBLE: np.array([1.0 if fresh else 0.0]),
    }


def resize_camera1_rgb(img: np.ndarray, size: int = CAMERA1_RGB_RESOLUTION) -> np.ndarray:
    """
    Resize a baseplate-camera frame to the policy grid, keeping the whole frame.

    No 4:3 crop (unlike ``resize_camera0_rgb``): the tag ring touches the frame's long edges, and a
    crop would lose tags. The aspect ratio changes (1280x800 -> 224x224); a policy learns on it the
    same way. INTER_AREA, as for camera0.
    """
    import cv2

    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img)
    return np.ascontiguousarray(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
