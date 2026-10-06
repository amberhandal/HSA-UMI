#!/usr/bin/env python3
"""
Check that the published ``fer_hand_tcp`` is the end effector the robot has configured -- ON THE STATION, at bringup.

The streaming controller converts O_T_EE onto ``hsa_tcp`` through ``fer_hand_tcp``, a static transform
this repo publishes that must equal the F_T_EE set in Desk; nothing else notices when it does not
(nuc/hsa_calib.py, "Where fer_hand_tcp comes from"). One-shot: read
``/franka_robot_state_broadcaster/robot_state`` once, compare ``f_t_ee`` with hsa_calib's transform, exit 0
on a match and 1 on a mismatch (bringup.launch.py aborts on 1). A message with no usable ``f_t_ee`` is
logged as "cannot verify" and exits 0. Self-contained, so it runs from a plain clone on the station:

    python3 nuc/check_ee_frame.py [timeout_s] [--calib hsa_calib] [--desk-end-effector none|hand]
"""

import argparse
import importlib
import math
from pathlib import Path
import sys
import threading

from franka_msgs.msg import FrankaRobotState
import rclpy
from rclpy.node import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOPIC = '/franka_robot_state_broadcaster/robot_state'
DEFAULT_TIMEOUT_S = 60.0
# Tolerances well under any real difference between the two candidate frames (10 cm, 45 deg)
# and well over the float noise of a quaternion round trip.
POSITION_TOL_M = 0.001
ANGLE_TOL_RAD = math.radians(0.5)


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """ZYX Euler angles to a unit quaternion ``(x, y, z, w)``, the tf2 convention."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def compare(position, quaternion, expected_xyz, expected_rpy) -> tuple[bool, str]:
    """
    Compare a reported pose with the expected transform; return ``(match, human-readable detail)``.

    :param position: reported translation ``(x, y, z)`` in metres.
    :param quaternion: reported rotation ``(x, y, z, w)``.
    :param expected_xyz: hsa_calib's translation for the Desk setting in force.
    :param expected_rpy: hsa_calib's roll/pitch/yaw for it.
    """
    dpos = math.sqrt(sum((p - e) ** 2 for p, e in zip(position, expected_xyz)))
    expected_q = quaternion_from_rpy(*expected_rpy)
    dot = abs(sum(a * b for a, b in zip(quaternion, expected_q)))
    angle = 2.0 * math.acos(min(1.0, dot))
    detail = (
        f'robot F_T_EE: xyz=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m, '
        f'{math.degrees(angle):.2f} deg from the published fer_hand_tcp; '
        f'translation differs by {dpos * 1e3:.1f} mm'
    )
    return dpos <= POSITION_TOL_M and angle <= ANGLE_TOL_RAD, detail


def read_f_t_ee(node: Node, timeout_s: float):
    """
    Return ``(position, quaternion)`` from the first robot_state message, or a string saying why not.

    The string outcomes are "timeout" (nothing published -- is the broadcaster spawned?) and
    "unavailable" (the message has no populated f_t_ee), which the caller reports differently.
    """
    got = threading.Event()
    box = {}

    def _cb(msg: FrankaRobotState) -> None:
        try:
            pose = msg.f_t_ee.pose
            q = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            # An all-zero quaternion is not a rotation; it is a field nobody filled in.
            if math.sqrt(sum(v * v for v in q)) < 0.5:
                box['why'] = 'unavailable'
            else:
                box['pose'] = ((pose.position.x, pose.position.y, pose.position.z), q)
        except AttributeError:
            box['why'] = 'unavailable'
        got.set()

    sub = node.create_subscription(FrankaRobotState, TOPIC, _cb, 10)
    deadline = node.get_clock().now().nanoseconds / 1e9 + timeout_s
    while not got.is_set() and node.get_clock().now().nanoseconds / 1e9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if 'pose' in box:
        return box['pose']
    return box.get('why', 'timeout')


def main(argv=None) -> int:
    """Run the check; return a shell exit code (0 match or unverifiable, 1 mismatch)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('timeout_s', nargs='?', type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument('--calib', default='hsa_calib', help='calibration module beside this script')
    parser.add_argument('--desk-end-effector', default=None, help="override the module's DESK_END_EFFECTOR")
    args = parser.parse_args(argv)
    calib = importlib.import_module(args.calib)
    setting = args.desk_end_effector or calib.DESK_END_EFFECTOR
    expected_xyz, expected_rpy = calib.ee_transform(setting)

    rclpy.init()
    node = rclpy.create_node('check_ee_frame')
    try:
        node.get_logger().info(
            f'[check_ee_frame] expecting Desk end effector {setting!r}: '
            f'{calib.TCP_PARENT} -> {calib.EE_FRAME} xyz={expected_xyz} rpy={expected_rpy}'
        )
        outcome = read_f_t_ee(node, args.timeout_s)
        if outcome == 'timeout':
            node.get_logger().error(
                f'[check_ee_frame] nothing on {TOPIC} within {args.timeout_s:.0f}s -- is bringup up with '
                'franka_robot_state_broadcaster spawned, and is FCI enabled?'
            )
            return 1
        if outcome == 'unavailable':
            node.get_logger().warning(
                f'[check_ee_frame] {TOPIC} carries no usable f_t_ee on this franka_ros2; CANNOT VERIFY '
                f'that Desk is set to {setting!r}. Check it by hand in Desk before execute_arm:=true.'
            )
            return 0
        position, quaternion = outcome
        ok, detail = compare(position, quaternion, expected_xyz, expected_rpy)
        if ok:
            node.get_logger().info(f'[check_ee_frame] match: {detail}')
            return 0
        others = sorted(k for k in calib.DESK_EE_TRANSFORMS if k != setting)
        node.get_logger().error(
            f'[check_ee_frame] MISMATCH: {detail}. The robot is not configured with the end effector '
            f'{setting!r}. Relaunch with desk_end_effector:={"|".join(others)}, or change it in Desk.'
        )
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # The launch is shutting down around us; 130 tells the gate this was an interrupt.
        sys.exit(130)
