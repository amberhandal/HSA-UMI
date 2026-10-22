"""
The measured constants of what HSA-UMI bolts to the FER's flange: the TCP frame, and the payload.

``hsa_tcp`` is the wrist's moving plate at its neutral servo setpoint, a fixed child of ``fer_link8``. Its
consumers -- TF (bringup.launch.py) and the streaming controller (``tcp_frame`` in
nuc/config/hsa_controllers.yaml) -- must agree, so the numbers live here and nowhere else. Values marked
ESTIMATE are placeholders: measure them, set ``CALIBRATION_STATUS`` to the date, then verify the TCP with
``ros2 run polyumi_ros2 tcp_pivot_test --ros-args -p eef_frame:=hsa_tcp`` and the payload by watching for
droop when the impedance controller activates (docs/calibration-instructions.md, "End-effector payload").
"""

import math
import os
from typing import NamedTuple

ARM_ID = 'fer'

# What is on the flange, from HSA_TOOL in the shell that launches bringup (nuc/station_env.sh passes it
# through): wrist = HSA wrist (default), rigid = metal base plate + rigid key mount, bare = nothing (TCP at the
# flange, zero payload).
TOOLS = ('wrist', 'rigid', 'bare')
TOOL = os.environ.get('HSA_TOOL', 'wrist')
if TOOL not in TOOLS:
    raise ValueError(f'HSA_TOOL must be one of {TOOLS}, not {TOOL!r}')
TCP_PARENT = f'{ARM_ID}_link8'
TCP_CHILD = 'hsa_tcp'
#: The frame franka reports O_T_EE in, as the streaming controller names it (``<arm_id>_hand_tcp``).
EE_FRAME = f'{ARM_ID}_hand_tcp'

#: 'measured YYYY-MM-DD' once the ESTIMATE values below have been replaced. Logged at bringup.
CALIBRATION_STATUS = {
    'bare': 'bare flange (HSA_TOOL=bare): TCP at the flange, no payload',
    'wrist': 'TCP z measured 2026-10-22 (key tip at 360 deg); payload CoM and extents ESTIMATE (nuc/hsa_calib.py)',
    'rigid': 'ESTIMATE (unmeasured placeholders -- see nuc/hsa_calib.py)',
}[TOOL]

# --------------------------------------------------------------------------------------------
# The TCP: the moving plate's tool surface, centred, at the neutral servo setpoint. ESTIMATE. Along
# the flange axis only: the base plate bolts to the flange's bolt circle, so x and y are 0 by symmetry.
# Flange face -> key tip along the flange axis, with every HSA at 360 deg (the demo pose; rest is ~12 mm shorter,
# camera_tracking.md). Measured 2026-10-22 with a rule, +/- 2 mm. Re-measure if the key, its mount or the
# extension angle changes.
WRIST_TCP_Z = 0.236
RIGID_TCP_Z = 0.150  # m, flange face -> key mount TCP. ESTIMATE: placeholder, measure and replace.
PLATE_HEIGHT_Z = {'wrist': WRIST_TCP_Z, 'rigid': RIGID_TCP_Z, 'bare': 0.0}[TOOL]

TCP_XYZ = (0.0, 0.0, PLATE_HEIGHT_Z)

# The yaw is a convention, not a measurement: only consumer agreement matters. -45 deg is franka's own
# hand-mount clocking, so hsa_tcp's x is where fer_hand's x would be.
TCP_RPY = (0.0, 0.0, -math.pi / 4)

# --------------------------------------------------------------------------------------------
# Where fer_hand_tcp comes from: the controller takes franka's O_T_EE as the pose of <arm_id>_hand_tcp
# and converts it onto tcp_frame through TF. With no hand in the URDF, bringup.launch.py publishes
# fer_link8 -> fer_hand_tcp statically, and it MUST equal the F_T_EE of the end effector configured in
# Desk: a wrong value fails nowhere, the spring is simply anchored 10 cm from where the policy thinks
# the plate is. check_ee_frame.py verifies it at bringup; the desk_end_effector launch argument
# overrides this per run.
DESK_END_EFFECTOR = 'none'

# fer_link8 -> fer_hand_tcp for each Desk setting, as (xyz, rpy). The hand's numbers are
# franka_description's own (fer_link8 -> fer_hand is Rz(-45 deg); fer_hand -> fer_hand_tcp is
# 0.1034 m along z), composed: a yaw leaves a pure z offset untouched.
DESK_EE_TRANSFORMS = {
    'none': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    'hand': ((0.0, 0.0, 0.1034), (0.0, 0.0, -math.pi / 4)),
}
# The end-effector mass Desk carries for each setting (robot_state inertia_ee.m). The FCI adds it to the SetLoad
# mass, so any Desk mass that is not in this table is compensated twice: with 1.08 kg entered in Desk and the
# full 1.018 kg wrist in SetLoad the arm reported a steady 9 N phantom force at rest. check_ee_frame.py refuses
# bringup when the robot's m_ee differs from this by more than DESK_EE_MASS_TOL.
DESK_EE_MASS = {'none': 0.0, 'hand': 0.73}
DESK_EE_MASS_TOL = 0.05  # kg

# --------------------------------------------------------------------------------------------
# The payload -- everything past the flange, for libfranka's gravity compensation. Unmodelled mass is
# a constant force the impedance spring fights (dz = m * g / K_trans the moment the controller
# activates). Pushed once at bringup via franka_msgs/srv/SetLoad (bringup.launch.py).
#
# The whole assembly, not a residual: with Desk's end effector at "None" the robot carries no m_ee
# of its own. With "Franka Hand" the FCI already assumes 0.73 kg at the hand's CoM and this must
# become the DIFFERENCE -- check with
#     ros2 topic echo /franka_robot_state_broadcaster/robot_state --field inertia_ee --once
#
# Two parts, each a uniform solid box on the flange axis, combined below into the one load SetLoad takes.
# Masses weighed 2026-10-05 (oz -> kg). Positions and sizes are ESTIMATES until measured.
OZ = 0.028349523125  # kg


class PayloadPart(NamedTuple):
    """One rigid piece past the flange: mass, CoM height above the flange face, box size (x, y, z)."""

    name: str
    mass: float  # kg
    com_z: float  # m, in fer_link8
    extents: tuple[float, float, float]  # m


#: Base plate, bolted to the flange. ESTIMATE thickness and footprint.
BASE_PLATE_THICKNESS = 0.008  # m
BASE_PLATE = PayloadPart('base plate', 3.0 * OZ, BASE_PLATE_THICKNESS / 2, (0.12, 0.12, BASE_PLATE_THICKNESS))

#: The ODrive HSA wrist with the key mount and key, weighed as one piece 2026-10-21: 933 g, without the base
#: plate (weighed separately, BASE_PLATE above). Its CoM is taken at mid-height, a uniform box from the base plate up
#: to the moving plate. ESTIMATE for the height and the CoM; the motors sit low on the base, so the true CoM is
#: probably below mid-height.
_WRIST_HEIGHT = PLATE_HEIGHT_Z - BASE_PLATE_THICKNESS
HSA_WRIST = PayloadPart(
    'HSA wrist + key', 0.933, BASE_PLATE_THICKNESS + _WRIST_HEIGHT / 2, (0.12, 0.12, max(_WRIST_HEIGHT, 0.01))
)

#: Rigid setup: a metal base plate (always on for demos) and the rigid key attachment, weighed 2026-10-13.
METAL_PLATE_THICKNESS = 0.006  # m. ESTIMATE.
METAL_BASE_PLATE = PayloadPart(
    'metal base plate', 0.100, METAL_PLATE_THICKNESS / 2, (0.12, 0.12, METAL_PLATE_THICKNESS)
)
_MOUNT_LENGTH = max(RIGID_TCP_Z - METAL_PLATE_THICKNESS, 0.01)
#: CoM at mid-length of the mount, a slender box along the flange axis. ESTIMATE.
KEY_MOUNT = PayloadPart(
    'rigid key mount', 0.079, METAL_PLATE_THICKNESS + _MOUNT_LENGTH / 2, (0.04, 0.04, _MOUNT_LENGTH)
)

PAYLOAD_PARTS: tuple[PayloadPart, ...] = {
    'wrist': (BASE_PLATE, HSA_WRIST),
    'rigid': (METAL_BASE_PLATE, KEY_MOUNT),
    'bare': (),
}[TOOL]

PAYLOAD_MASS = sum(part.mass for part in PAYLOAD_PARTS)  # kg

# In fer_link8 directly (there is no hand frame to convert from); every part is centred on the flange axis.
PAYLOAD_COM_FLANGE = (
    0.0,
    0.0,
    sum(part.mass * part.com_z for part in PAYLOAD_PARTS) / PAYLOAD_MASS if PAYLOAD_MASS else 0.0,
)


def _stp_args(xyz, rpy, parent, child) -> list[str]:
    """tf2_ros static_transform_publisher argv for one fixed transform."""
    x, y, z = xyz
    roll, pitch, yaw = rpy
    return [
        '--x', str(x), '--y', str(y), '--z', str(z),
        '--roll', str(roll), '--pitch', str(pitch), '--yaw', str(yaw),
        '--frame-id', parent, '--child-frame-id', child,
    ]  # fmt: skip


def static_transform_publisher_args() -> list[str]:
    """Build the tf2_ros static_transform_publisher argv publishing TCP_PARENT -> TCP_CHILD."""
    return _stp_args(TCP_XYZ, TCP_RPY, TCP_PARENT, TCP_CHILD)


def ee_transform(desk_end_effector: str | None = None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Return ``(xyz, rpy)`` of ``fer_link8 -> fer_hand_tcp`` for a Desk end-effector setting.

    :param desk_end_effector: a key of :data:`DESK_EE_TRANSFORMS`; defaults to :data:`DESK_END_EFFECTOR`.
    :raises ValueError: on a setting this file does not know, rather than guessing identity.
    """
    setting = DESK_END_EFFECTOR if desk_end_effector is None else desk_end_effector
    try:
        return DESK_EE_TRANSFORMS[setting]
    except KeyError:
        raise ValueError(
            f'unknown Desk end effector {setting!r}; expected one of {sorted(DESK_EE_TRANSFORMS)}'
        ) from None


def ee_transform_publisher_args(desk_end_effector: str | None = None) -> list[str]:
    """Build the static_transform_publisher argv for ``fer_link8 -> fer_hand_tcp``."""
    xyz, rpy = ee_transform(desk_end_effector)
    return _stp_args(xyz, rpy, TCP_PARENT, EE_FRAME)


def payload_com_flange() -> tuple[float, float, float]:
    """
    Return the payload's centre of mass in ``fer_link8``, the frame SetLoad's F_x_Cload is in.

    Already stated in that frame, so this is the identity -- kept as a function so set_payload.py
    reads every calibration module through the same three names.
    """
    return PAYLOAD_COM_FLANGE


def payload_inertia_flange() -> tuple[float, ...]:
    """
    Build the payload's inertia tensor about its combined CoM, in ``fer_link8``, column-major.

    Each part is a uniform solid box in the flange's axes, moved to the combined CoM with the parallel-axis
    theorem; all CoMs lie on the z axis, so the tensor stays diagonal. Computed rather than written down so it
    cannot go stale against the masses. The FER rejects a nonzero mass carrying a zero tensor, and a sum of
    real boxes always satisfies the triangle inequality the firmware checks.
    """
    zc = PAYLOAD_COM_FLANGE[2]
    ixx = iyy = izz = 0.0
    for part in PAYLOAD_PARTS:
        w, d, h = part.extents
        k = part.mass / 12.0
        shift = part.mass * (part.com_z - zc) ** 2
        ixx += k * (d**2 + h**2) + shift
        iyy += k * (w**2 + h**2) + shift
        izz += k * (w**2 + d**2)
    return (ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)


def set_load_request() -> str:
    """Build the franka_msgs/srv/SetLoad request literal for a `ros2 service call`."""
    com = ', '.join(str(v) for v in payload_com_flange())
    inertia = ', '.join(str(v) for v in payload_inertia_flange())
    return f'{{mass: {PAYLOAD_MASS}, center_of_mass: [{com}], load_inertia: [{inertia}]}}'


def describe() -> str:
    """One-line summary of the TCP in force, logged at bringup so it is never a mystery."""
    return (
        f'tool={TOOL}; {TCP_PARENT} -> {TCP_CHILD}: xyz={TCP_XYZ} rpy={TCP_RPY}; '
        f'calibration: {CALIBRATION_STATUS} (nuc/hsa_calib.py)'
    )


def describe_payload() -> str:
    """One-line summary of the payload pushed to the FCI, for the same reason as :func:`describe`."""
    parts = ', '.join(f'{part.name} {part.mass:.3f} kg' for part in PAYLOAD_PARTS) or 'nothing'
    return (
        f'payload: {PAYLOAD_MASS:.3f} kg ({parts}), CoM {payload_com_flange()} in {TCP_PARENT}; '
        f'calibration: {CALIBRATION_STATUS} (nuc/hsa_calib.py)'
    )
