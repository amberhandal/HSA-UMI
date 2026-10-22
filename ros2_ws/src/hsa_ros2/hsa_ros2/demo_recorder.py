"""
Record teleoperated demonstrations into a UMI ReplayBuffer, one stored step per camera frame.

The camera frame is the clock: the TF pose, the teleop target (the ``action``), the stage and the
optional wrench, wrist state, plate pose and wrist camera streams are all sampled at the frame's
capture instant (``stamp - latency.camera``), the pairing ``policy_client_node`` uses at inference.
The wrist streams (``wrist``, ``plate``, ``wrist_camera``) come from the wrist_control workspace
(``ros2 launch wrist_control wrist_control.launch.py``); ``wrist.enabled`` needs its
``hsa_wrist_msgs`` package sourced, which is why the import is inside the branch. The
``/hsa_recorder/*`` Trigger services are the ones ``spacemouse_teleop`` calls.

    ros2 run hsa_ros2 demo_recorder --ros-args -p output_dir:=~/hsa_data/peg_v1 --params-file .../recorder.yaml
    ros2 service call /hsa_recorder/start_episode std_srvs/srv/Trigger "{}"   # or press c in the teleop
    ros2 service call /hsa_recorder/stop_episode  std_srvs/srv/Trigger "{}"   # or s
    ros2 service call /hsa_recorder/export_zip    std_srvs/srv/Trigger "{}"   # -> <output_dir>/<name>.zarr.zip
"""

from __future__ import annotations

import csv
import pathlib
import threading
import json
import time
from collections import deque

import numpy as np
import rclpy
import rclpy.time
import scipy.spatial.transform as st
import tf2_ros
from geometry_msgs.msg import PoseStamped, WrenchStamped
from trajectory_msgs.msg import MultiDOFJointTrajectory
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException  # type: ignore[attr-defined]

# The pixel contract, shared with policy_client_node: crop to 4:3 then INTER_AREA resize to 224.
from polyumi_ros2.camera_preproc import resize_camera0_rgb

from hsa_ros2 import dataset_writer as dw
from hsa_ros2.episode_video import EpisodeVideo, delete_episode_video


def _stamp_s(stamp) -> float:
    """builtin_interfaces/Time -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class EpisodePlan:
    """
    The per-episode schedule: an optional CSV whose rows are handed out one per episode.

    The recorder never interprets the columns; every column of the current row is written verbatim
    into that episode's sidecar (no file: only the running index). Rows advance when an episode is
    SAVED (a dropped episode keeps its row), the teleop's 'n' key skips one, and the plan wraps around.
    """

    def __init__(self, path: str):
        """Load the plan from ``path``."""
        self.path = path
        self.rows: list[dict[str, str]] = []
        self.index = 0
        if path:
            with open(pathlib.Path(path).expanduser(), newline='') as f:
                self.rows = [{k.strip(): (v or '').strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
            if not self.rows:
                raise ValueError(f'plan {path} has a header but no rows')

    @property
    def current(self) -> dict[str, str]:
        """The row for the episode about to be recorded ({} without a plan file)."""
        return self.rows[self.index % len(self.rows)] if self.rows else {}

    def describe(self) -> str:
        """Return a description of the current entry."""
        if not self.rows:
            return f'episode index {self.index}, no plan file'
        wrapped = ' (plan wrapped around)' if self.index >= len(self.rows) else ''
        fields = ', '.join(f'{k}={v}' for k, v in self.current.items())
        return f'plan row {self.index % len(self.rows) + 1}/{len(self.rows)}{wrapped}: {fields}'

    def advance(self) -> str:
        """Advance to the next entry."""
        self.index += 1
        return self.describe()


class DemoRecorder(Node):
    """Sample every stream on the camera grid and append episodes to a ReplayBuffer on disk."""

    def __init__(self) -> None:
        """Set up the node."""
        super().__init__('hsa_recorder')

        # --- Parameters ------------------------------------------------------------------
        self.declare_parameter('output_dir', '~/hsa_data/session')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('base_frame', 'fer_link0')
        self.declare_parameter('tcp_frame', 'hsa_tcp')
        # Camera photon -> header.stamp delay. Measure once with `latency_probe mode:=camera` and keep
        # equal to inference.yaml's latency.gopro so train and deploy agree.
        self.declare_parameter('latency.camera', 0.0)
        self.declare_parameter('latency.proprio', 0.001)
        # 1 stores every frame (30 Hz with the RealSense at 30 fps -> obs_down_sample_steps: 3 in
        # the task config). 3 stores at 10 Hz -> obs_down_sample_steps: 1.
        self.declare_parameter('store_every_n_frames', 1)
        self.declare_parameter('teleop_target_topic', '/hsa/teleop_target')
        # Where the stored `action` comes from: teleop (demos) or policy (rollouts: the command chunks the
        # policy client sends the controller, each waypoint keyed on the time it is due).
        self.declare_parameter('action_source', 'teleop')
        self.declare_parameter('policy_target_topic', '/polyumi/target_poses_traj')
        # A full-resolution mp4 per episode in <output_dir>/videos (every camera frame, with its stamp).
        self.declare_parameter('video.enabled', False)
        self.declare_parameter('video.fps', 30.0)
        self.declare_parameter('stage_topic', '/hsa/stage')
        self.declare_parameter('stage_names', list(dw.DEFAULT_STAGE_NAMES))
        # The ODrive HSA wrist: hsa_wrist_msgs/WristState from wrist_state_publisher (sourced from the
        # wrist_control workspace), the newest message at or before each frame's capture instant.
        self.declare_parameter('wrist.enabled', False)
        self.declare_parameter('wrist.topic', '/hsa/wrist/state')
        # The top plate from the baseplate camera (camera_tracking platform_pose): the pose in its rest
        # frame, held when the tags are not seen; plate_visible says whether the stored pose is fresh.
        self.declare_parameter('plate.enabled', False)
        self.declare_parameter('plate.topic', '/platform_pose')
        self.declare_parameter('plate.visible_topic', '/platform_tag_visible')
        self.declare_parameter('plate.max_age_s', 0.2)
        # The baseplate camera frame itself, resized to the policy grid as camera1_rgb. The raw topic
        # or its /compressed republication (less DDS traffic, a JPEG decode per stored frame).
        self.declare_parameter('wrist_camera.enabled', False)
        self.declare_parameter('wrist_camera.topic', '/tracking_cam/image_raw')
        self.declare_parameter('wrist_camera.max_age_s', 0.2)
        self.declare_parameter('wrench.enabled', False)
        self.declare_parameter('wrench.topic', '/franka_robot_state_broadcaster/external_wrench_in_base_frame')
        # A target older than this at a frame's instant means the teleop node is not running or
        # has stalled; the frame is still stored (with the held target) but a warning says so.
        self.declare_parameter('max_target_age_s', 0.5)
        self.declare_parameter('buffers.seconds', 2.0)
        # UMI's sampler draws zero samples from an episode shorter than (16-1)*down_sample+1 steps.
        self.declare_parameter('min_steps', 46)
        # A demo still recording after this many seconds is saved as a failure (stopped_by: timeout). 0 = no limit.
        self.declare_parameter('max_episode_s', 0.0)
        # Provenance written to the per-episode sidecar JSON.
        self.declare_parameter('task_name', '')
        # Per-episode labels, both written to the sidecar: `plan` is a CSV with any columns, one row per
        # episode (see EpisodePlan); `condition` is a free-form tag read afresh at every episode start:
        #     ros2 param set /hsa_recorder condition 'wrist=articulated stiffness=soft'
        self.declare_parameter('plan', '')
        self.declare_parameter('condition', '')
        self.declare_parameter('advance_plan_on_save', True)
        self.declare_parameter('wrist_set', '')
        self.declare_parameter('wrist_hours', 0.0)
        self.declare_parameter('notes', '')

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._output_dir = pathlib.Path(str(p('output_dir'))).expanduser()
        self._base = str(p('base_frame'))
        self._tcp = str(p('tcp_frame'))
        self._lat_cam = float(p('latency.camera'))
        self._lat_proprio = float(p('latency.proprio'))
        self._every_n = max(1, int(p('store_every_n_frames')))
        self._wrist_on = bool(p('wrist.enabled'))
        self._plate_on = bool(p('plate.enabled'))
        self._plate_max_age = float(p('plate.max_age_s'))
        self._wcam_on = bool(p('wrist_camera.enabled'))
        self._wcam_max_age = float(p('wrist_camera.max_age_s'))
        self._wrench_on = bool(p('wrench.enabled'))
        self._max_target_age = float(p('max_target_age_s'))
        self._min_steps = int(p('min_steps'))
        self._max_episode_s = float(p('max_episode_s'))
        buffer_s = float(p('buffers.seconds'))
        self._plan = EpisodePlan(str(p('plan')))
        self._advance_on_save = bool(p('advance_plan_on_save'))

        # --- The store ---------------------------------------------------------------------
        self._writer = dw.ReplayBufferWriter(
            self._output_dir,
            attrs={
                'source': 'hsa_ros2.demo_recorder',
                'base_frame': self._base,
                'tcp_frame': self._tcp,
                'latency_camera_s': self._lat_cam,
                'latency_proprio_s': self._lat_proprio,
                'store_every_n_frames': self._every_n,
                'image_size': dw.IMAGE_SIZE,
                'action_key': 'commanded teleop target [pos, rotvec, gripper]',
                # Note: no gripper yet, just a placeholder.
                'gripper_closed_width_m': 0.0,
                'task_name': str(p('task_name')),
                'wrist_state_topic': str(p('wrist.topic')) if self._wrist_on else '',
                'plate_pose_topic': str(p('plate.topic')) if self._plate_on else '',
                'wrist_camera_topic': str(p('wrist_camera.topic')) if self._wcam_on else '',
                'camera1_resize': 'resize_wrist_rgb: whole frame to 224x224, INTER_AREA' if self._wcam_on else '',
            },
        )

        # --- Live state ----------------------------------------------------------------------
        self._lock = threading.Lock()
        self._recording = False
        self._steps: list[dict] = []
        self._episode_t0: float | None = None
        self._wrench_ages: list[float] = []
        self._wrench_missing = 0
        self._last_wrench_age: float | None = None
        # Per-episode pairing ages of the wrist streams, for the sidecar; the last one for the status.
        self._ages: dict[str, list[float]] = {'wrist': [], 'plate': [], 'wrist_camera': []}
        self._missing: dict[str, int] = {'wrist': 0, 'plate': 0, 'wrist_camera': 0}
        self._last_age: dict[str, float | None] = {'wrist': None, 'plate': None, 'wrist_camera': None}
        self._plate_visible_steps = 0
        self._frame_count = 0
        self._tf_fail = 0
        self._stale_target = 0
        self._last_image_stamp: float | None = None
        self._latest_stage = 0
        # (stamp, sample) rings, newest last, for time alignment at the frame instant.
        maxlen = int(buffer_s * 200)
        self._targets: deque = deque(maxlen=maxlen)  # (receive_time, pose6)
        self._wrench: deque = deque(maxlen=maxlen)  # (stamp, wrench6)
        self._wrist: deque = deque(maxlen=maxlen)  # (stamp, {key: (6,) array})
        self._plate: deque = deque(maxlen=maxlen)  # (stamp, pose6)
        self._plate_visible = False
        # Raw baseplate frames are 3 MB each, so only a short window of them is kept.
        self._wcam: deque = deque(maxlen=max(4, int(buffer_s * 15)))  # (stamp, rgb HxWx3)

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=buffer_s))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- Subscriptions -------------------------------------------------------------------
        self.create_subscription(Image, str(p('image_topic')), self._image_cb, 1)
        self._action_source = str(p('action_source'))
        if self._action_source == 'policy':
            self.create_subscription(
                MultiDOFJointTrajectory, str(p('policy_target_topic')), self._policy_chunk_cb, 10
            )
        elif self._action_source == 'teleop':
            self.create_subscription(PoseStamped, str(p('teleop_target_topic')), self._target_cb, 10)
        else:
            raise ValueError(f"action_source must be 'teleop' or 'policy', not {self._action_source!r}")
        self._video = (
            EpisodeVideo(self._output_dir / 'videos', float(p('video.fps'))) if bool(p('video.enabled')) else None
        )
        self.create_subscription(Int32, str(p('stage_topic')), self._stage_cb, 10)
        if self._wrist_on:
            try:
                from hsa_wrist_msgs.msg import WristState
            except ImportError as e:
                raise RuntimeError(
                    'wrist.enabled needs hsa_wrist_msgs: source the wrist_control workspace '
                    '(~/hsa-ws/wrist_control/odrive_ros_ws/install/setup.bash) in this terminal'
                ) from e
            self.create_subscription(WristState, str(p('wrist.topic')), self._wrist_cb, 20)
        if self._plate_on:
            self.create_subscription(PoseStamped, str(p('plate.topic')), self._plate_cb, 20)
            self.create_subscription(Bool, str(p('plate.visible_topic')), self._plate_visible_cb, 20)
        if self._wcam_on:
            topic = str(p('wrist_camera.topic'))
            if topic.endswith('/compressed'):
                self.create_subscription(CompressedImage, topic, self._wcam_compressed_cb, 1)
            else:
                self.create_subscription(Image, topic, self._wcam_cb, 1)
        if self._wrench_on:
            self.create_subscription(WrenchStamped, str(p('wrench.topic')), self._wrench_cb, 20)

        # --- Services --------------------------------------------------------------------------
        # Names are absolute so the teleop's clients find them whatever this node is called.
        for name, cb in (
            ('start_episode', self._srv_start),
            ('stop_episode', self._srv_stop),
            ('stop_failure', self._srv_stop_failure),
            ('drop_episode', self._srv_drop),
            ('next', self._srv_next),
            ('export_zip', self._srv_export),
        ):
            self.create_service(Trigger, f'/hsa_recorder/{name}', cb)

        self.create_timer(5.0, self._status)
        self._status_pub = self.create_publisher(String, '/hsa_recorder/status', 10)
        self.create_timer(0.1, self._publish_status)
        self.create_timer(0.1, self._check_timeout)
        self.get_logger().info(
            f'hsa_recorder: {self._writer.summary().splitlines()[0]} | image {p("image_topic")} every '
            f'{self._every_n} frame(s) | TF {self._base} -> {self._tcp} @ stamp - {self._lat_cam}s | '
            f'wrench={"on" if self._wrench_on else "off"} wrist={"on" if self._wrist_on else "off"} '
            f'plate={"on" if self._plate_on else "off"} wrist_camera={"on" if self._wcam_on else "off"} | '
            f'{self._plan.describe()}'
        )

    # ------------------------------------------------------------------------------------------
    # Stream callbacks: cache with stamps; only the image callback creates steps
    # ------------------------------------------------------------------------------------------

    def _target_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        pose6 = np.concatenate(
            [
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                st.Rotation.from_quat([q.x, q.y, q.z, q.w]).as_rotvec(),
            ]
        )
        # Keyed on receive time, not the header stamp: the teleop stamps a target with the instant
        # it is due (one tick ahead), while "the command in force at t" is the newest one that had
        # been issued by t.
        self._targets.append((self.get_clock().now().nanoseconds * 1e-9, pose6))

    def _policy_chunk_cb(self, msg: MultiDOFJointTrajectory) -> None:
        """Store a policy chunk's waypoints by due time, replacing the not-yet-due rest of the previous one."""
        t0 = _stamp_s(msg.header.stamp)
        new = []
        for pt in msg.points:
            if not pt.transforms:
                continue
            tf = pt.transforms[0]
            q = tf.rotation
            pose6 = np.concatenate(
                [[tf.translation.x, tf.translation.y, tf.translation.z],
                 st.Rotation.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()]
            )  # fmt: skip
            new.append((t0 + pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9, pose6))
        if not new:
            return
        # The controller splices a new chunk in at its first waypoint, so keep the times sorted.
        while self._targets and self._targets[-1][0] >= new[0][0]:
            self._targets.pop()
        self._targets.extend(new)

    def _stage_cb(self, msg: Int32) -> None:
        self._latest_stage = int(msg.data)

    def _wrist_cb(self, msg) -> None:
        self._wrist.append((_stamp_s(msg.header.stamp), dw.wrist_state_arrays(msg)))

    def _plate_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        pose6 = np.concatenate(
            [
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                st.Rotation.from_quat([q.x, q.y, q.z, q.w]).as_rotvec(),
            ]
        )
        self._plate.append((_stamp_s(msg.header.stamp), pose6))

    def _plate_visible_cb(self, msg: Bool) -> None:
        self._plate_visible = bool(msg.data)

    def _wcam_cb(self, msg: Image) -> None:
        if msg.encoding not in ('rgb8', 'bgr8', 'mono8'):
            self.get_logger().error(f'wrist camera: unsupported encoding {msg.encoding!r}', throttle_duration_sec=5.0)
            return
        if msg.encoding == 'mono8':
            img = np.repeat(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 1), 3, axis=2)
        else:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == 'bgr8':
                img = img[:, :, ::-1]
        self._wcam.append((_stamp_s(msg.header.stamp), img))

    def _wcam_compressed_cb(self, msg: CompressedImage) -> None:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().error('wrist camera: could not decode a compressed frame', throttle_duration_sec=5.0)
            return
        self._wcam.append((_stamp_s(msg.header.stamp), bgr[:, :, ::-1]))

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        w = msg.wrench
        self._wrench.append(
            (
                _stamp_s(msg.header.stamp),
                np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]),
            )
        )

    def _image_cb(self, msg: Image) -> None:
        self._last_image_stamp = _stamp_s(msg.header.stamp)
        self._frame_count += 1
        with self._lock:
            recording = self._recording
        if recording and self._video is not None:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self._video.write(frame[:, :, ::-1] if msg.encoding.lower() == 'bgr8' else frame, _stamp_s(msg.header.stamp))
        if not recording or (self._frame_count % self._every_n):
            return
        step = self._sample_step(msg)
        if step is not None:
            with self._lock:
                if self._recording:
                    self._steps.append(step)

    # ------------------------------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------------------------------

    def _sample_step(self, msg: Image) -> dict | None:
        """Build one stored step from a camera frame, or None (counted) if a stream is missing."""
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().error(f'unsupported image encoding {msg.encoding!r}', throttle_duration_sec=5.0)
            return None
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        image = resize_camera0_rgb(np.ascontiguousarray(img))

        # The instant this frame was captured; everything below is sampled there.
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        t_capture = stamp - Duration(seconds=self._lat_cam)
        t_s = t_capture.nanoseconds * 1e-9

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base, self._tcp, t_capture + Duration(seconds=self._lat_proprio)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self._tf_fail += 1
            self.get_logger().warning(f'frame skipped, TF lookup failed: {e}', throttle_duration_sec=2.0)
            return None
        tr, ro = tf.transform.translation, tf.transform.rotation
        eef_pos = np.array([tr.x, tr.y, tr.z])
        eef_rot = st.Rotation.from_quat([ro.x, ro.y, ro.z, ro.w]).as_rotvec()

        targets = list(self._targets)
        target = dw.latest_before([t for t, _ in targets], [v for _, v in targets], t_s) if targets else None
        if target is None:
            self.get_logger().warning(
                f'frame skipped, no {self._action_source} target received yet', throttle_duration_sec=2.0
            )
            return None
        if targets and t_s - targets[-1][0] > self._max_target_age:
            self._stale_target += 1
            self.get_logger().warning(
                f'newest teleop target is {t_s - targets[-1][0]:.2f}s old at this frame; is spacemouse_teleop running?',
                throttle_duration_sec=2.0,
            )

        step = {
            dw.KEY_IMAGE: image,
            dw.KEY_POS: eef_pos,
            dw.KEY_ROT: eef_rot,
            dw.KEY_GRIPPER: np.zeros(1),
            dw.KEY_ACTION: np.concatenate([target, [0.0]]),
            dw.KEY_STAGE: np.array([self._latest_stage]),
            dw.KEY_TIME: np.array(t_s),
        }
        if self._wrist_on:
            sample, age = self._held('wrist', self._wrist, t_s)
            if sample is None:
                # Publisher not up yet: the episode would lack the wrist columns entirely, so skip.
                self.get_logger().warning('frame skipped, no wrist state yet (wrist_state_publisher?)', throttle_duration_sec=2.0)
                return None
            step.update(sample)
        if self._plate_on:
            pose6, age = self._held('plate', self._plate, t_s)
            fresh = pose6 is not None and age <= self._plate_max_age and self._plate_visible
            if pose6 is None:
                pose6 = np.full(6, np.nan)
                self.get_logger().warning(
                    'no plate pose yet: zero the platform (set_platform_zero) with the tags in view', throttle_duration_sec=5.0
                )
            step[dw.KEY_PLATE_POS] = pose6[:3]
            step[dw.KEY_PLATE_ROT] = pose6[3:]
            step[dw.KEY_PLATE_VISIBLE] = np.array([1 if fresh else 0])
            self._plate_visible_steps += int(fresh)
        if self._wcam_on:
            frame, age = self._held('wrist_camera', self._wcam, t_s)
            if frame is None:
                self.get_logger().warning('frame skipped, no wrist camera frame yet', throttle_duration_sec=2.0)
                return None
            if age > self._wcam_max_age:
                self.get_logger().warning(f'wrist camera frame is {age:.2f}s old at this step', throttle_duration_sec=2.0)
            step[dw.KEY_WRIST_IMAGE] = dw.resize_wrist_rgb(frame)
        if self._wrench_on:
            wr = list(self._wrench)
            times = np.array([w[0] for w in wr], dtype=np.float64)
            i = int(np.searchsorted(times, t_s, side='right')) - 1 if wr else -1
            if i < 0:
                step[dw.KEY_WRENCH] = np.full(6, np.nan)
                self._wrench_missing += 1
            else:
                step[dw.KEY_WRENCH] = wr[i][1]
                self._last_wrench_age = t_s - float(times[i])
                self._wrench_ages.append(self._last_wrench_age)
        return step

    def _held(self, name: str, ring: deque, t_s: float):
        """Newest sample of ``ring`` stamped at or before ``t_s`` (zero-order hold) and its age; counted."""
        items = list(ring)
        times = np.array([x[0] for x in items], dtype=np.float64)
        i = int(np.searchsorted(times, t_s, side='right')) - 1 if items else -1
        if i < 0:
            self._missing[name] += 1
            return None, None
        age = t_s - float(times[i])
        self._last_age[name] = age
        self._ages[name].append(age)
        return items[i][1], age

    # ------------------------------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------------------------------

    def _srv_start(self, req, resp):
        with self._lock:
            if self._recording:
                resp.success, resp.message = False, 'already recording'
                return resp
        if self._last_image_stamp is None:
            resp.success, resp.message = False, 'no camera frame received yet; is the camera driver up?'
            return resp
        try:
            self._tf_buffer.lookup_transform(self._base, self._tcp, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            resp.success, resp.message = False, f'TF {self._base} -> {self._tcp} unavailable: {e}'
            return resp
        if self._video is not None:
            self._video.discard()  # a partial file left by an interrupted episode
        with self._lock:
            self._steps = []
            self._recording = True
            self._episode_t0 = time.time()
            self._tf_fail = self._stale_target = self._wrench_missing = 0
            self._wrench_ages = []
            self._ages = {k: [] for k in self._ages}
            self._missing = dict.fromkeys(self._missing, 0)
            self._plate_visible_steps = 0
        resp.success = True
        resp.message = f'recording episode {self._writer.n_episodes}: {self._plan.describe()}' + (
            f' | condition: {self._condition()}' if self._condition() else ''
        )
        self.get_logger().info(resp.message)
        return resp

    def _srv_stop(self, req, resp):
        return self._stop(resp, 'success')

    def _srv_stop_failure(self, req, resp):
        """Save the episode labelled a failure: kept for analysis, left out of the training export."""
        return self._stop(resp, 'failure')

    def _stop(self, resp, outcome: str, stopped_by: str = 'operator'):
        with self._lock:
            if not self._recording:
                resp.success, resp.message = False, 'not recording'
                return resp
            self._recording = False
            steps, t0 = self._steps, self._episode_t0
            self._steps = []
        if not steps:
            if self._video is not None:
                self._video.discard()
            resp.success, resp.message = False, 'episode had no stored steps; nothing saved'
            self.get_logger().error(resp.message)
            return resp
        arrays = dw.assemble_episode(steps)
        idx = self._writer.add_episode(arrays)
        video = self._video.finish(f'episode_{idx:04d}') if self._video is not None else None
        duration = float(arrays[dw.KEY_TIME][-1] - arrays[dw.KEY_TIME][0])
        self._writer.write_sidecar(
            idx,
            {
                'episode': idx,
                'outcome': outcome,
                'stopped_by': stopped_by,
                'action_source': self._action_source,
                'video': str(video) if video else None,
                'n_steps': len(steps),
                'duration_s': duration,
                'started_at': t0,
                'stored_at': dw.now_iso(),
                'plan_index': self._plan.index,
                'plan': self._plan.current,
                'plan_file': self._plan.path,
                'condition': self._condition(),
                'task_name': self.get_parameter('task_name').value,
                'wrist_set': self.get_parameter('wrist_set').value,
                'wrist_hours': self.get_parameter('wrist_hours').value,
                'notes': self.get_parameter('notes').value,
                'tf_lookup_failures': self._tf_fail,
                'stale_target_frames': self._stale_target,
                'max_stage': int(arrays[dw.KEY_STAGE].max()),
                'stage_names': list(self.get_parameter('stage_names').value),
                'stages': self._stage_table(arrays),
                'wrench_sample_age_s': self._wrench_age_stats(),
                'wrist_sample_age_s': self._age_stats('wrist') if self._wrist_on else None,
                'plate_sample_age_s': self._age_stats('plate') if self._plate_on else None,
                'plate_visible_fraction': (self._plate_visible_steps / len(steps)) if self._plate_on else None,
                'wrist_camera_age_s': self._age_stats('wrist_camera') if self._wcam_on else None,
                'wrist_extension_deg_median': (
                    np.nanmedian(arrays[dw.KEY_WRIST_EXT], axis=0).round(1).tolist() if self._wrist_on else None
                ),
                'wrist_command_deg_median': (
                    np.nanmedian(arrays[dw.KEY_WRIST_CMD], axis=0).round(1).tolist() if self._wrist_on else None
                ),
                'wrist_iq_a_median': (
                    np.nanmedian(arrays[dw.KEY_WRIST_IQ], axis=0).round(2).tolist() if self._wrist_on else None
                ),
                'keys': sorted(arrays),
            },
        )
        short = (
            ''
            if len(steps) >= self._min_steps
            else f' (SHORT: under {self._min_steps} steps, the sampler will draw nothing from it)'
        )
        resp.success = True
        how = ' after the time limit' if stopped_by == 'timeout' else ''
        resp.message = f'saved episode {idx} ({outcome.upper()}{how}): {len(steps)} steps, {duration:.1f}s, {self._tf_fail} TF drops{short}'
        if self._advance_on_save:
            resp.message += f' | next: {self._plan.advance()}'
        self.get_logger().info(resp.message)
        return resp

    def _check_timeout(self) -> None:
        """Save a demo that has run past max_episode_s as a failure, as if f had been pressed."""
        if self._max_episode_s <= 0:
            return
        with self._lock:
            t0 = self._episode_t0 if self._recording else None
        if t0 is None or time.time() - t0 < self._max_episode_s:
            return
        self.get_logger().warning(f'demo reached the {self._max_episode_s:g} s limit: saving it as a FAILURE')
        self._stop(Trigger.Response(), 'failure', stopped_by='timeout')

    def _srv_drop(self, req, resp):
        with self._lock:
            if self._recording:
                n = len(self._steps)
                self._recording, self._steps = False, []
                if self._video is not None:
                    self._video.discard()
                resp.success, resp.message = True, f'discarded the episode in progress ({n} steps)'
                self.get_logger().warning(resp.message)
                return resp
        idx = self._writer.drop_last_episode()
        if idx is None:
            resp.success, resp.message = False, 'nothing to drop: not recording and the store is empty'
        else:
            if self._video is not None:
                delete_episode_video(self._video.dir, f'episode_{idx:04d}')
            resp.success, resp.message = True, f'dropped saved episode {idx} (store now has {self._writer.n_episodes})'
            self.get_logger().warning(resp.message)
        return resp

    def _srv_next(self, req, resp):
        resp.success, resp.message = True, f'skipped to {self._plan.advance()}'
        self.get_logger().info(resp.message)
        return resp

    def _condition(self) -> str:
        """Return the free-form condition tag, read live so `ros2 param set` applies to the next episode."""
        return str(self.get_parameter('condition').value)

    def _srv_export(self, req, resp):
        with self._lock:
            if self._recording:
                resp.success, resp.message = False, 'stop the episode before exporting'
                return resp
        if self._writer.n_episodes == 0:
            resp.success, resp.message = False, 'store is empty'
            return resp
        # Failures are kept in the store for analysis but never trained on.
        outcomes = self._writer.episode_outcomes()
        keep = [i for i, o in enumerate(outcomes) if o != 'failure']
        if not keep:
            resp.success, resp.message = False, 'every episode is a failure; nothing to export'
            return resp
        out = self._writer.export_zip(episodes=keep)
        resp.success, resp.message = (
            True,
            f'exported {out} ({len(keep)} episodes; {len(outcomes) - len(keep)} failures left out)',
        )
        self.get_logger().info(resp.message)
        return resp

    # ------------------------------------------------------------------------------------------

    def _stage_table(self, arrays: dict) -> list[dict]:
        """One row per contiguous subtask run: index, name, step range and duration."""
        names = list(self.get_parameter('stage_names').value)
        t = arrays[dw.KEY_TIME].reshape(-1)
        return [
            {
                'stage': value,
                'name': names[value] if 0 <= value < len(names) else None,
                'start_step': a,
                'end_step': b,
                'duration_s': float(t[b - 1] - t[a]),
            }
            for value, a, b in dw.stage_segments(arrays[dw.KEY_STAGE])
        ]

    def _wrench_age_stats(self) -> dict | None:
        """Age of the wrench sample paired with each step: how far the force lagged the frame."""
        if not self._wrench_on:
            return None
        ages = np.array(self._wrench_ages, dtype=np.float64)
        return {
            'median': float(np.median(ages)) if ages.size else None,
            'max': float(ages.max()) if ages.size else None,
            'missing_steps': int(self._wrench_missing),
        }

    def _age_stats(self, name: str) -> dict:
        """Age of the sample paired with each step for one wrist stream (sidecar)."""
        ages = np.array(self._ages[name], dtype=np.float64)
        return {
            'median': float(np.median(ages)) if ages.size else None,
            'max': float(ages.max()) if ages.size else None,
            'missing_steps': int(self._missing[name]),
        }

    def _wrist_status(self) -> dict:
        """Live wrist / plate / wrist-camera fields for collect_ui: the newest samples, not paired ones."""
        out: dict = {}
        ms = lambda a: None if a is None else a * 1e3  # noqa: E731
        if self._wrist_on:
            latest = self._wrist[-1] if self._wrist else None
            out['wrist_ext_deg'] = None if latest is None else [round(float(v), 1) for v in latest[1][dw.KEY_WRIST_EXT]]
            out['wrist_iq_a'] = None if latest is None else [round(float(v), 2) for v in latest[1][dw.KEY_WRIST_IQ]]
            out['wrist_errors'] = None if latest is None else [int(v) for v in latest[1][dw.KEY_WRIST_ERRORS]]
            out['wrist_age_ms'] = ms(self._last_age['wrist'])
        if self._plate_on:
            latest = self._plate[-1] if self._plate else None
            out['plate_pos_mm'] = None if latest is None else [round(float(v) * 1e3, 2) for v in latest[1][:3]]
            out['plate_tilt_deg'] = None if latest is None else round(float(np.degrees(np.linalg.norm(latest[1][3:]))), 2)
            out['plate_visible'] = bool(self._plate_visible)
            out['plate_age_ms'] = ms(self._last_age['plate'])
        if self._wcam_on:
            out['wrist_camera_age_ms'] = ms(self._last_age['wrist_camera'])
        return out

    def _publish_status(self) -> None:
        """Publish the recorder state as JSON for collect_ui (10 Hz)."""
        with self._lock:
            rec, n, t0 = self._recording, len(self._steps), self._episode_t0
        self._status_pub.publish(
            String(
                data=json.dumps(
                    {
                        'episode': self._writer.n_episodes,
                        'recording': rec,
                        'elapsed_s': (time.time() - t0) if (rec and t0) else 0.0,
                        'max_episode_s': self._max_episode_s,
                        't0': t0 if rec else None,
                        'steps': n,
                        'store_episodes': self._writer.n_episodes,
                        'store_steps': self._writer.n_steps,
                        'wrench_age_ms': None if self._last_wrench_age is None else self._last_wrench_age * 1e3,
                        'condition': self._condition(),
                        'plan': self._plan.describe(),
                        **self._wrist_status(),
                    }
                )
            )
        )

    def _status(self) -> None:
        with self._lock:
            rec, n = self._recording, len(self._steps)
        age = None if self._last_image_stamp is None else time.time() - self._last_image_stamp
        self.get_logger().info(
            f'{"REC " + str(n) + " steps" if rec else "idle"} | store {self._writer.n_episodes} ep / '
            f'{self._writer.n_steps} steps | last frame {"none" if age is None else f"{age:.2f}s ago"} | '
            f'targets {len(self._targets)} wrench {len(self._wrench)} wrist {len(self._wrist)} '
            f'plate {len(self._plate)}{" (visible)" if self._plate_visible else ""} wrist_cam {len(self._wcam)}'
        )


def main() -> None:
    """Run the recorder node."""
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = DemoRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
