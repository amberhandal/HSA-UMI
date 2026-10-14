"""
Live view for demo collection: the camera image with the demo number, force and timestamp on it.

    ros2 run hsa_ros2 collect_ui --ros-args -p latency.camera:=0.03

Keys in the window: c start an episode, s save it, f save it as a failure, backspace twice within 2 s drop it,
q close the view. The force shown is the wrench sample the recorder pairs with that frame (the
newest sample stamped at or before the frame's capture instant) and its age is printed next to it,
so the pairing is visible while collecting. Timestamps are the frame's capture instant.
"""

from __future__ import annotations

import json
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

FONT = cv2.FONT_HERSHEY_SIMPLEX
RECORDER_SERVICES = ('start_episode', 'stop_episode', 'stop_failure', 'drop_episode')


def pair_force(times, wrenches, t: float) -> tuple[np.ndarray | None, float | None]:
    """Return (wrench, age_s) for the newest sample stamped at or before ``t``, or (None, None)."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return None, None
    i = int(np.searchsorted(times, t, side='right')) - 1
    if i < 0:
        return None, None
    return np.asarray(wrenches[i], dtype=np.float64), float(t - times[i])


def overlay_lines(status: dict, force: np.ndarray | None, force_age_s: float | None, t_capture_s: float, stage: int):
    """Compose the text lines: demo number and state, force, timestamp, stage."""
    episode = status.get('episode')
    recording = bool(status.get('recording'))
    first = f'Demo {episode if episode is not None else "?"}'
    if recording:
        first += f'   REC {float(status.get("elapsed_s", 0.0)):.1f} s   {int(status.get("steps", 0))} steps'
    else:
        first += '   idle'
    if force is None:
        force_line = 'Force   no data'
    else:
        force_line = (
            f'Force {float(np.linalg.norm(force[:3])):5.1f} N   '
            f'({force[0]:+.1f} {force[1]:+.1f} {force[2]:+.1f})   sample {float(force_age_s) * 1e3:.0f} ms old'
        )
    stamp = time.strftime('%H:%M:%S', time.localtime(t_capture_s)) + f'.{int((t_capture_s % 1.0) * 1000):03d}'
    if recording and status.get('t0'):
        stamp += f'   t = {t_capture_s - float(status["t0"]):.3f} s'
    return [first, force_line, stamp, f'Stage {stage}']


def draw_overlay(bgr: np.ndarray, lines: list[str], recording: bool) -> np.ndarray:
    """Draw the lines top-left in white on a black outline, and a red dot while recording."""
    out = bgr.copy()
    for i, line in enumerate(lines):
        y = 30 + 28 * i
        cv2.putText(out, line, (12, y), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if recording:
        cv2.circle(out, (out.shape[1] - 30, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def image_to_bgr(msg: Image) -> np.ndarray | None:
    """Convert an rgb8 or bgr8 Image to a BGR array; other encodings return None."""
    if msg.encoding not in ('rgb8', 'bgr8'):
        return None
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else arr.copy()


class CollectUI(Node):
    """Show the camera with the recorder's state, the paired force and the capture timestamp."""

    def __init__(self) -> None:
        """Set up the window, the subscriptions and the recorder service clients."""
        super().__init__('collect_ui')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('wrench.topic', '/franka_robot_state_broadcaster/external_wrench_in_base_frame')
        self.declare_parameter('status_topic', '/hsa_recorder/status')
        self.declare_parameter('stage_topic', '/hsa/stage')
        self.declare_parameter('recorder_service_prefix', '/hsa_recorder')
        self.declare_parameter('latency.camera', 0.0)  # same value as the recorder's
        self.declare_parameter('buffer_seconds', 2.0)
        self.declare_parameter('window', 'hsa collect')
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._lat_cam = float(p('latency.camera'))
        self._window = str(p('window'))
        self._frame: np.ndarray | None = None
        self._t_capture: float | None = None
        self._wrench: deque = deque(maxlen=int(float(p('buffer_seconds')) * 1000))
        self._status: dict = {}
        self._stage = 0
        self._last_backspace = 0.0
        self.create_subscription(Image, str(p('image_topic')), self._image_cb, 1)
        self.create_subscription(WrenchStamped, str(p('wrench.topic')), self._wrench_cb, 20)
        self.create_subscription(String, str(p('status_topic')), self._status_cb, 10)
        self.create_subscription(Int32, str(p('stage_topic')), self._stage_cb, 10)
        prefix = str(p('recorder_service_prefix'))
        self._recorder_clients = {name: self.create_client(Trigger, f'{prefix}/{name}') for name in RECORDER_SERVICES}
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        self.create_timer(1.0 / 30.0, self._tick)
        self.get_logger().info(f'collect_ui: {p("image_topic")} | force {p("wrench.topic")} | keys c s f backspace q')

    def _image_cb(self, msg: Image) -> None:
        bgr = image_to_bgr(msg)
        if bgr is None:
            self.get_logger().warning(f'unsupported image encoding {msg.encoding}', throttle_duration_sec=5.0)
            return
        self._frame = bgr
        self._t_capture = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9 - self._lat_cam

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        w = msg.wrench
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self._wrench.append((stamp, (w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z)))

    def _status_cb(self, msg: String) -> None:
        try:
            self._status = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning('recorder status is not JSON', throttle_duration_sec=5.0)

    def _stage_cb(self, msg: Int32) -> None:
        self._stage = int(msg.data)

    def _tick(self) -> None:
        if self._frame is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, 'waiting for camera frames', (12, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            wr = list(self._wrench)
            force, age = pair_force([s for s, _ in wr], [w for _, w in wr], self._t_capture)
            lines = overlay_lines(self._status, force, age, self._t_capture, self._stage)
            img = draw_overlay(self._frame, lines, bool(self._status.get('recording')))
        cv2.imshow(self._window, img)
        key = cv2.pollKey()
        if key == -1:
            return
        key &= 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info('quit')
            if rclpy.ok():
                rclpy.shutdown()
        elif key == ord('c'):
            self._call('start_episode')
        elif key == ord('s'):
            self._call('stop_episode')
        elif key == ord('f'):
            self._call('stop_failure')
        elif key in (8, 127):
            now = time.monotonic()
            if now - self._last_backspace < 2.0:
                self._last_backspace = 0.0
                self._call('drop_episode')
            else:
                self._last_backspace = now
                self.get_logger().info('backspace again within 2 s to drop the episode')

    def _call(self, name: str) -> None:
        client = self._recorder_clients[name]
        if not client.service_is_ready():
            self.get_logger().warning(f'{client.srv_name} not available: is demo_recorder running?')
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.get_logger().info(f'{name}: {f.result().message}')
            if f.result() is not None
            else self.get_logger().error(f'{name}: call failed')
        )


def main() -> None:
    """Run the view until q is pressed or ROS shuts down."""
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = CollectUI()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
