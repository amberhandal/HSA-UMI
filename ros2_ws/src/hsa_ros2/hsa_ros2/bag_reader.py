"""
Read a rollout bag (scripts/record_rollout.sh) into plain arrays for the figures.

Predicted chunks come from the trajectory topic, the executed path from TF replayed through a
tf2 buffer, the client's diagnostics from ``/polyumi/diag/*``, the teleop target and the external
wrench from their topics.
"""

from __future__ import annotations

import pathlib

import numpy as np
import rclpy.time
import rosbag2_py
import tf2_ros
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

TRAJ_TOPIC = '/polyumi/target_poses_traj'
DIAG_PREFIX = '/polyumi/diag/'
TARGET_TOPIC = '/hsa/teleop_target'
WRENCH_TOPIC = '/franka_robot_state_broadcaster/external_wrench_in_base_frame'


def _stamp_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def read_messages(path, topics: list[str] | None = None):
    """Yield ``(topic, message, t_s)`` for every message in the bag, in time order."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=''), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, deserialize_message(data, types[topic]), t_ns * 1e-9


class Rollout:
    """Everything the rollout figures need, loaded from one bag directory."""

    def __init__(self, path, *, base_frame: str = 'fer_link0', tcp_frame: str = 'hsa_tcp', sample_hz: float = 50.0):
        """Load the bag at ``path`` and replay its TF into a buffer."""
        self.path = pathlib.Path(path)
        self.chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self.diag: dict[str, tuple[list[float], list[float]]] = {}
        self.target_times: list[float] = []
        self.target_xyz: list[np.ndarray] = []
        self.wrench_times: list[float] = []
        self.wrench_norm: list[float] = []
        tf_msgs: list[tuple[str, object]] = []
        t_first = t_last = None
        for topic, msg, t_s in read_messages(self.path):
            t_first = t_s if t_first is None else t_first
            t_last = t_s
            if topic == TRAJ_TOPIC:
                anchor = _stamp_s(msg.header.stamp)
                times, xyz = [], []
                for pt in msg.points:
                    if pt.transforms:
                        tr = pt.transforms[0].translation
                        times.append(anchor + _stamp_s(pt.time_from_start))
                        xyz.append((tr.x, tr.y, tr.z))
                if times:
                    self.chunks.append((np.array(times), np.array(xyz)))
            elif topic.startswith(DIAG_PREFIX):
                self.diag.setdefault(topic[len(DIAG_PREFIX) :], ([], []))
                self.diag[topic[len(DIAG_PREFIX) :]][0].append(t_s)
                self.diag[topic[len(DIAG_PREFIX) :]][1].append(float(msg.data))
            elif topic == TARGET_TOPIC:
                p = msg.pose.position
                self.target_times.append(_stamp_s(msg.header.stamp))
                self.target_xyz.append(np.array((p.x, p.y, p.z)))
            elif topic == WRENCH_TOPIC:
                f = msg.wrench.force
                self.wrench_times.append(_stamp_s(msg.header.stamp))
                self.wrench_norm.append(float(np.hypot(np.hypot(f.x, f.y), f.z)))
            elif topic in ('/tf', '/tf_static'):
                tf_msgs.append((topic, msg))
        self.executed_times = np.zeros(0)
        self.executed_xyz = np.zeros((0, 3))
        if tf_msgs and t_first is not None:
            buf = tf2_ros.Buffer(cache_time=Duration(seconds=(t_last - t_first) + 10.0))
            for topic, msg in tf_msgs:
                for tr in msg.transforms:
                    (buf.set_transform_static if topic == '/tf_static' else buf.set_transform)(tr, 'bag')
            times, xyz = [], []
            for t in np.arange(t_first, t_last, 1.0 / sample_hz):
                try:
                    tr = buf.lookup_transform(base_frame, tcp_frame, rclpy.time.Time(seconds=t)).transform.translation
                except tf2_ros.TransformException:
                    continue
                times.append(t)
                xyz.append((tr.x, tr.y, tr.z))
            self.executed_times = np.array(times)
            self.executed_xyz = np.array(xyz).reshape(-1, 3)

    def tracking_error(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(times, error_m)`` of the executed path against what was commanded.

        The reference is the teleop target when the bag has one, otherwise the chunk the controller
        was following at each instant.
        """
        from hsa_ros2.figures import commanded_at

        times, err = [], []
        tt = np.array(self.target_times)
        for t, p in zip(self.executed_times, self.executed_xyz):
            if tt.size:
                i = int(np.searchsorted(tt, t, side='right')) - 1
                ref = self.target_xyz[i] if i >= 0 else None
            else:
                ref = commanded_at(self.chunks, t)
            if ref is not None:
                times.append(t)
                err.append(float(np.linalg.norm(p - ref)))
        return np.array(times), np.array(err)

    def summary(self) -> str:
        """One line: what the bag held."""
        return (
            f'{self.path.name}: {len(self.chunks)} chunks, {len(self.executed_times)} executed samples, '
            f'{len(self.target_times)} teleop targets, {len(self.wrench_times)} wrench samples, '
            f'diag {sorted(self.diag)}'
        )
