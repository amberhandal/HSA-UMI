"""
The measured constants of what HSA-UMI bolts to the FER's flange: the TCP frame, and the payload.

``hsa_tcp`` is the wrist's moving plate at its neutral servo setpoint, a fixed child of ``fer_link8``. Its
consumers -- TF (fer_bringup.launch.py) and the streaming controller (``tcp_frame`` in
nuc/config/hsa_controllers.yaml) -- must agree, so the numbers live here and nowhere else. Values marked
ESTIMATE are placeholders: measure them, set ``CALIBRATION_STATUS`` to the date, then verify the TCP with
``ros2 run polyumi_ros2 tcp_pivot_test --ros-args -p eef_frame:=hsa_tcp`` and the payload by watching for
droop when the impedance controller activates (docs/calibration-instructions.md, "End-effector payload").
"""

import math
import os

ARM_ID = 'fer'

# HSA_TOOL=bare: nothing on the flange, so the TCP is the flange itself and the payload is zero. Set it
# in the shell that launches bringup (nuc/station_env.sh passes it through); the default is the wrist.
TOOL = os.environ.get('HSA_TOOL', 'wrist')
if TOOL not in ('wrist', 'bare'):
    raise ValueError(f"HSA_TOOL must be 'wrist' or 'bare', not {TOOL!r}")
TCP_PARENT = f'{ARM_ID}_link8'
TCP_CHILD = 'hsa_tcp'
#: The frame franka reports O_T_EE in, as the streaming controller names it (``<arm_id>_hand_tcp``).
EE_FRAME = f'{ARM_ID}_hand_tcp'

#: 'measured YYYY-MM-DD' once the ESTIMATE values below have been replaced. Logged at bringup.
CALIBRATION_STATUS = (
    'bare flange (HSA_TOOL=bare): TCP at the flange, no payload'
    if TOOL == 'bare'
    else 'ESTIMATE (unmeasured placeholders -- see nuc/hsa_calib.py)'
)

# --------------------------------------------------------------------------------------------
# The TCP: the moving plate's tool surface, centred, at the neutral servo setpoint. ESTIMATE. Along
# the flange axis only: the base plate bolts to the flange's bolt circle, so x and y are 0 by symmetry.
PLATE_HEIGHT_Z = 0.0 if TOOL == 'bare' else 0.150  # m, flange face -> plate tool surface, neutral servos

TCP_XYZ = (0.0, 0.0, PLATE_HEIGHT_Z)

# The yaw is a convention, not a measurement: only consumer agreement matters. -45 deg is franka's own
# hand-mount clocking, so hsa_tcp's x is where fer_hand's x would be.
TCP_RPY = (0.0, 0.0, -math.pi / 4)

# --------------------------------------------------------------------------------------------
# Where fer_hand_tcp comes from: the controller takes franka's O_T_EE as the pose of <arm_id>_hand_tcp
# and converts it onto tcp_frame through TF. With no hand in the URDF, fer_bringup.launch.py publishes
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

# --------------------------------------------------------------------------------------------
# The payload -- everything past the flange, for libfranka's gravity compensation. Unmodelled mass is
# a constant force the impedance spring fights (dz = m * g / K_trans the moment the controller
# activates). Pushed once at bringup via franka_msgs/srv/SetLoad (fer_bringup.launch.py).
#
# The whole assembly, not a residual: with Desk's end effector at "None" the robot carries no m_ee
# of its own. With "Franka Hand" the FCI already assumes 0.73 kg at the hand's CoM and this must
# become the DIFFERENCE -- check with
#     ros2 topic echo /franka_robot_state_broadcaster/robot_state --field inertia_ee --once
#
# ESTIMATE: six Dynamixel XM430-W350 at 82 g (0.49 kg) plus two plates, six HSAs, the U2D2 and
# the cabling the arm carries. Replace with a scale reading of everything past the flange.
PAYLOAD_MASS = 0.0 if TOOL == 'bare' else 0.9  # kg

# In fer_link8 directly (there is no hand frame to convert from): the servos sit on the base
# plate, so the CoM is below mid-height. ESTIMATE.
PAYLOAD_COM_FLANGE = (0.0, 0.0, 0.0) if TOOL == 'bare' else (0.0, 0.0, 0.06)  # m

# The FER firmware rejects a nonzero mass carrying a zero inertia ("invalid argument"), so this cannot
# be left at zero. A uniform solid box centred on the CoM, in fer_link8 axes, ordered (x, y, z): a real
# box's principal moments always satisfy the triangle inequality the firmware checks. ESTIMATE; inertia
# only enters the acceleration terms and the arm moves slowly, so the order of magnitude is what matters.
PAYLOAD_EXTENTS = (0.12, 0.12, 0.15)


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
    Build the payload's inertia tensor about its CoM, in ``fer_link8``, column-major.

    A uniform solid box from :data:`PAYLOAD_EXTENTS`, whose axes are already the flange's, so the
    tensor is diagonal. Computed rather than written down so it cannot go stale against
    :data:`PAYLOAD_MASS`; the FER rejects a nonzero mass carrying a zero tensor.
    """
    w, d, h = PAYLOAD_EXTENTS
    k = PAYLOAD_MASS / 12.0
    ixx, iyy, izz = k * (d**2 + h**2), k * (w**2 + h**2), k * (w**2 + d**2)
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
    return (
        f'payload: {PAYLOAD_MASS} kg, CoM {payload_com_flange()} in {TCP_PARENT}; '
        f'calibration: {CALIBRATION_STATUS} (nuc/hsa_calib.py)'
    )
