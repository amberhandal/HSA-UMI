"""
SpaceMouse teleoperation of the Franka TCP through PolyUMI's streaming impedance controller.

Integrates the target pose and publishes it as a small *action chunk* -- the message the streaming Cartesian
impedance controller on the arm computer already understands, and the same message the policy
client publishes at inference. Teleop and policy drive the arm through one identical path.

How it communicates (every arrow is a ROS topic or service; names are parameters):

    SpaceMouse (USB HID) --pyspacemouse, 200 Hz thread--> this node
    TF  fer_link0 -> polyumi_tcp   --lookup, once at start / on 'r'----> seed of the target pose
    this node --/polyumi/target_poses_traj (MultiDOFJointTrajectory)---> streaming controller (execute: true)
    this node --/polyumi/target_poses_preview (PoseArray)--------------> Foxglove/RViz only (execute: false)
    this node --/hsa/teleop_target (PoseStamped)-----------------------> recorder (becomes the `action` key)
    this node --/hsa/target_servo (JointState)-------------------------> wrist command node (optional)
    this node --/hsa/stage (Int32)-------------------------------------> recorder
    keyboard  --c/s/backspace/n--> std_srvs/Trigger services on the recorder
                                   (/hsa_recorder/start_episode, stop_episode, drop_episode, next_placement)

Run it in its OWN terminal (keyboard input needs a TTY):

    ros2 run hsa_ros2 spacemouse_teleop --ros-args --params-file $(ros2 pkg prefix hsa_ros2)/share/hsa_ros2/config/teleop.yaml
    ros2 run hsa_ros2 spacemouse_teleop --ros-args -p execute:=true ...       # MOVES THE ARM
    ros2 run hsa_ros2 spacemouse_teleop --ros-args -p input_source:=fake_sine # desk test, no device

Keys: c start episode, s stop, Backspace drop, Space stage+1, n next placement,
p print pose, r re-seed target from TF, x toggle wrist stiff/soft, q quit.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException  # type: ignore[attr-defined]

# PolyUMI's wiring: the topic the controller listens on, and the preview format.  Importing from
# polyumi_ros2 (not the generic client package) keeps the topic name defined in exactly one place.
from polyumi_ros2.target_chunk import CONSUMER_HINT, TargetChunkPublisher, pose_array

from hsa_ros2.teleop_math import (
    clamp_workspace,
    fake_sine_axes,
    integrate_target,
    map_spacemouse_axes,
    pose6_to_quat,
    quat_to_rotvec,
)

#: The wrist servo joint names, in bus-ID order.  The command node and the recorder use the same.
WRIST_JOINTS = [f'hsa_{i}' for i in range(1, 7)]


# ----------------------------------------------------------------------------------------------
# Input sources.  Each has one job: `read()` returns (axes[6] in [-1,1], buttons[2] as bools).
# ----------------------------------------------------------------------------------------------


class SpacemouseSource:
    """
    Reads the SpaceMouse in a background thread and hands out the latest state.

    pyspacemouse's ``read()`` consumes ONE USB report per call, and the device sends ~200 of them
    per second.  Reading only at the 10 Hz control rate would leave a growing backlog and the arm
    would respond to inputs from seconds ago.  So a thread drains the device continuously (as your
    ``Spacemouse`` class did in a separate process) and the control loop just takes a snapshot.
    """

    def __init__(self, deadzone: float):
        import pyspacemouse  # imported here so the fake/none sources work on machines without it

        self._device = pyspacemouse.open()
        if not self._device:
            raise RuntimeError('pyspacemouse.open() failed: is the SpaceMouse plugged in and the hidraw udev rule installed?')
        self._deadzone = deadzone
        self._lock = threading.Lock()
        self._axes = np.zeros(6)
        self._buttons = [False, False]
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name='spacemouse_reader', daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop:
            s = self._device.read()
            raw = [s.x, s.y, s.z, s.roll, s.pitch, s.yaw]
            buttons = list(getattr(s, 'buttons', []) or [])
            with self._lock:
                self._axes = map_spacemouse_axes(raw, self._deadzone)
                self._buttons = [bool(buttons[i]) if i < len(buttons) else False for i in range(2)]
            time.sleep(0.005)

    def read(self):
        with self._lock:
            return self._axes.copy(), list(self._buttons)

    def close(self) -> None:
        self._stop = True
        try:
            self._device.close()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass


class FakeSineSource:
    """Desk-test stand-in: a slow circle, no buttons.  See teleop_math.fake_sine_axes."""

    def __init__(self, period_s: float, amplitude: float):
        self._t0 = time.monotonic()
        self._period = period_s
        self._amp = amplitude

    def read(self):
        return fake_sine_axes(time.monotonic() - self._t0, self._period, self._amp), [False, True]

    def close(self) -> None:
        pass


class NoInputSource:
    """Keyboard-only session: the target never moves, but every topic/service hook still works."""

    def read(self):
        return np.zeros(6), [False, False]

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------------------------
# Keyboard: one character at a time from the terminal, without waiting for Enter.
# ----------------------------------------------------------------------------------------------


class KeyReader:
    """
    Puts every keypress on a queue from a background thread.

    Uses the terminal's *cbreak* mode (characters arrive immediately) and restores the terminal
    on close.  If stdin is not a terminal -- e.g. the node was started by `ros2 launch` -- the
    reader disables itself and says so, rather than crashing.
    """

    def __init__(self, logger):
        self.keys: queue.Queue[str] = queue.Queue()
        self._enabled = sys.stdin.isatty()
        self._old_settings = None
        if not self._enabled:
            logger.warning('stdin is not a TTY: keyboard commands disabled (run the node with `ros2 run` in a terminal).')
            return
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._stop = False
        threading.Thread(target=self._loop, name='key_reader', daemon=True).start()

    def _loop(self) -> None:
        while not self._stop:
            ch = sys.stdin.read(1)
            if ch:
                self.keys.put(ch)

    def close(self) -> None:
        self._stop = True
        if self._old_settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


# ----------------------------------------------------------------------------------------------
# The node.
# ----------------------------------------------------------------------------------------------


class SpacemouseTeleop(Node):
    """Integrate a target pose from the SpaceMouse and publish it as action chunks."""

    def __init__(self) -> None:
        super().__init__('spacemouse_teleop')

        # --- Parameters -------------------------------------------------------------------
        # declare_parameter(name, default) registers a setting that can be overridden from a yaml
        # file (--params-file), the command line (-p name:=value) or a launch file.  The default
        # here is what you get if nobody overrides it.  Dotted names group related settings.
        self.declare_parameter('control_hz', 10.0)
        self.declare_parameter('base_frame', 'fer_link0')
        self.declare_parameter('tcp_frame', 'polyumi_tcp')
        # false = publish the preview only (nothing moves).  true = publish chunks to the
        # controller topic.  Deliberately not toggleable from the keyboard.
        self.declare_parameter('execute', False)
        # Waypoints per tick.  1 reproduces the old scripts exactly (one target at now+dt).  2 adds
        # a copy at now+2dt so a late tick leaves the controller holding the target instead of
        # running out of trajectory.
        self.declare_parameter('chunk_len', 2)
        self.declare_parameter('max_pos_speed', 0.20)   # m/s at full deflection (old default)
        self.declare_parameter('max_rot_speed', 0.15)   # rad/s at full deflection (old default)
        # Workspace box in the base frame; the old WORKSPACE_X/Y/Z constants.
        self.declare_parameter('workspace.x_min', 0.2)
        self.declare_parameter('workspace.x_max', 0.9)
        self.declare_parameter('workspace.y_min', -0.4)
        self.declare_parameter('workspace.y_max', 0.4)
        self.declare_parameter('workspace.z_min', 0.05)
        self.declare_parameter('workspace.z_max', 0.5)
        # 'spacemouse' | 'fake_sine' | 'none'
        self.declare_parameter('input_source', 'spacemouse')
        self.declare_parameter('spacemouse.deadzone', 0.0)
        self.declare_parameter('spacemouse.rotation_button', 0)   # held -> rotate, translation locked
        self.declare_parameter('spacemouse.z_button', 1)          # held -> vertical motion allowed
        self.declare_parameter('fake_sine.period_s', 6.0)
        self.declare_parameter('fake_sine.amplitude', 0.4)
        # Wrist: publish a servo setpoint every tick so the command node keeps holding it.
        self.declare_parameter('wrist.enabled', False)
        self.declare_parameter('wrist.setpoint_rad', [0.0] * 6)
        self.declare_parameter('wrist.stiff_setpoint_rad', [0.0] * 6)
        # Where the recorder's services and the stage topic live.
        self.declare_parameter('recorder_service_prefix', '/hsa_recorder')
        self.declare_parameter('teleop_target_topic', '/hsa/teleop_target')
        self.declare_parameter('stage_topic', '/hsa/stage')
        self.declare_parameter('wrist_target_topic', '/hsa/target_servo')
        self.declare_parameter('preview_topic', '/polyumi/target_poses_preview')

        p = lambda name: self.get_parameter(name).value  # noqa: E731 - short accessor
        self._hz = float(p('control_hz'))
        self._dt = 1.0 / self._hz
        self._base = str(p('base_frame'))
        self._tcp = str(p('tcp_frame'))
        self._execute = bool(p('execute'))
        self._chunk_len = max(1, int(p('chunk_len')))
        self._max_pos_speed = float(p('max_pos_speed'))
        self._max_rot_speed = float(p('max_rot_speed'))
        self._box = {k: float(p(f'workspace.{k}')) for k in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')}
        self._rotation_button = int(p('spacemouse.rotation_button'))
        self._z_button = int(p('spacemouse.z_button'))
        self._wrist_enabled = bool(p('wrist.enabled'))
        self._wrist_soft = np.asarray(p('wrist.setpoint_rad'), dtype=np.float64)
        self._wrist_stiff = np.asarray(p('wrist.stiff_setpoint_rad'), dtype=np.float64)
        self._wrist_is_stiff = False

        # --- Input ------------------------------------------------------------------------
        source = str(p('input_source'))
        if source == 'spacemouse':
            self._input = SpacemouseSource(float(p('spacemouse.deadzone')))
        elif source == 'fake_sine':
            self._input = FakeSineSource(float(p('fake_sine.period_s')), float(p('fake_sine.amplitude')))
        elif source == 'none':
            self._input = NoInputSource()
        else:
            raise ValueError(f"input_source must be spacemouse | fake_sine | none, got '{source}'")
        self._keys = KeyReader(self.get_logger())

        # --- TF: where is the TCP right now? ----------------------------------------------
        # A Buffer stores every transform heard on /tf and /tf_static; the listener fills it.  We
        # only need the latest base->tcp transform, to seed the target where the arm actually is,
        # so a re-seed never yanks the arm somewhere else.
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._target: np.ndarray | None = None   # [x, y, z, rx, ry, rz], base frame

        # --- Publishers -------------------------------------------------------------------
        # create_publisher(MessageType, topic, queue_depth).  Publishing where nobody listens is
        # legal and silent, which is why the execute path checks the subscriber count each tick.
        self._chunk_pub = None
        if self._execute:
            self._chunk_pub = TargetChunkPublisher(self, frame_id=self._base, joint_name=self._tcp)
        self._preview_pub = self.create_publisher(PoseArray, str(p('preview_topic')), 10)
        self._target_pub = self.create_publisher(PoseStamped, str(p('teleop_target_topic')), 10)
        self._stage_pub = self.create_publisher(Int32, str(p('stage_topic')), 10)
        self._wrist_pub = self.create_publisher(JointState, str(p('wrist_target_topic')), 10) if self._wrist_enabled else None
        self._stage = 0

        # --- Service clients on the recorder ----------------------------------------------
        # std_srvs/Trigger is the simplest service type: an empty request, and a response with
        # `success` and `message`.  Calls are asynchronous: call_async() returns a Future and we
        # log the answer in a callback, so the control loop never blocks on the recorder.
        prefix = str(p('recorder_service_prefix'))
        self._recorder = {
            name: self.create_client(Trigger, f'{prefix}/{name}')
            for name in ('start_episode', 'stop_episode', 'drop_episode', 'next_placement')
        }

        # --- Control timer ----------------------------------------------------------------
        # create_timer(period, callback): rclpy calls _tick every 1/hz seconds from spin().
        self._tick_count = 0
        self.create_timer(self._dt, self._tick)

        mode = f'EXECUTE via {self._chunk_pub.topic_name} (ARM WILL MOVE)' if self._execute else 'preview only (nothing moves)'
        self.get_logger().info(
            f'spacemouse_teleop: {mode} | input={source} | {self._hz:g} Hz | '
            f'{self._max_pos_speed} m/s, {self._max_rot_speed} rad/s | frames {self._base} -> {self._tcp}'
        )
        self.get_logger().info('keys: c start  s stop  Backspace drop  Space stage+1  n next placement  p pose  r reseed  x wrist  q quit')

    # ------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------

    def _lookup_tcp(self) -> np.ndarray | None:
        """Latest base->tcp transform as a 6-vector, or None (with a throttled warning) if TF is not up."""
        try:
            tf = self._tf_buffer.lookup_transform(self._base, self._tcp, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warning(
                f'TF {self._base} -> {self._tcp} not available yet ({type(e).__name__}); '
                'is the arm bringup running and publishing polyumi_tcp?',
                throttle_duration_sec=2.0,
            )
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return np.concatenate([[t.x, t.y, t.z], quat_to_rotvec(r.x, r.y, r.z, r.w)])

    def _pose_msg(self, pose6: np.ndarray) -> Pose:
        """6-vector -> geometry_msgs/Pose (position + quaternion)."""
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = (float(v) for v in pose6[:3])
        q = pose6_to_quat(pose6)
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (float(v) for v in q)
        return msg

    def _call_recorder(self, name: str) -> None:
        """Fire a Trigger call at the recorder if the service exists; otherwise say so."""
        client = self._recorder[name]
        if not client.service_is_ready():
            self.get_logger().warning(f'recorder service {client.srv_name} is not available (recorder not running?)')
            return
        future = client.call_async(Trigger.Request())

        def done(fut):
            try:
                resp = fut.result()
                (self.get_logger().info if resp.success else self.get_logger().error)(f'{name}: {resp.message}')
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'{name} failed: {e!r}')

        future.add_done_callback(done)

    def _handle_key(self, ch: str) -> bool:
        """Dispatch one keypress.  Returns False when the node should quit."""
        if ch == 'q':
            return False
        if ch == 'c':
            self._stage = 0
            self._call_recorder('start_episode')
        elif ch == 's':
            self._call_recorder('stop_episode')
        elif ch in ('\x7f', '\x08'):  # Backspace
            self._call_recorder('drop_episode')
        elif ch == ' ':
            self._stage += 1
            self.get_logger().info(f'stage -> {self._stage}')
        elif ch == 'n':
            self._call_recorder('next_placement')
        elif ch == 'p':
            meas = self._lookup_tcp()
            self.get_logger().info(f'target {np.round(self._target, 4).tolist() if self._target is not None else None}')
            self.get_logger().info(f'measured {np.round(meas, 4).tolist() if meas is not None else None}')
        elif ch == 'r':
            self._target = None   # next tick re-seeds from TF
            self.get_logger().info('target will be re-seeded from TF')
        elif ch == 'x':
            self._wrist_is_stiff = not self._wrist_is_stiff
            self.get_logger().info(f'wrist -> {"STIFF" if self._wrist_is_stiff else "soft"} setpoint')
        return True

    # ------------------------------------------------------------------------------------------
    # The control tick: read -> integrate -> clamp -> publish.
    # ------------------------------------------------------------------------------------------

    def _tick(self) -> None:
        # 1. keyboard
        while not self._keys.keys.empty():
            if not self._handle_key(self._keys.keys.get_nowait()):
                self.get_logger().info('quit')
                rclpy.shutdown()
                return

        # 2. seed the target from wherever the arm is, the first time and after 'r'
        if self._target is None:
            seed = self._lookup_tcp()
            if seed is None:
                return
            self._target = clamp_workspace(seed, self._box)
            self.get_logger().info(f'target seeded from TF: {np.round(self._target, 4).tolist()}')

        # 3. read the input and advance the target by one tick
        axes, buttons = self._input.read()
        rotation_mode = bool(buttons[self._rotation_button]) if self._rotation_button < len(buttons) else False
        z_enabled = bool(buttons[self._z_button]) if self._z_button < len(buttons) else False
        self._target = clamp_workspace(
            integrate_target(
                self._target, axes,
                max_pos_speed=self._max_pos_speed, max_rot_speed=self._max_rot_speed, hz=self._hz,
                rotation_mode=rotation_mode, z_enabled=z_enabled,
            ),
            self._box,
        )

        # 4. publish.  Waypoint k of the chunk is due at now + (1 + k) * dt: the first one a full
        #    tick ahead (as the old scripts did), the copies after it just hold the target.
        now = self.get_clock().now()
        pose = self._pose_msg(self._target)
        if self._chunk_pub is not None:
            if self._chunk_pub.get_subscription_count() == 0:
                self.get_logger().warning(
                    f'nothing is subscribed to {self._chunk_pub.topic_name}: the arm will not move. Needs {CONSUMER_HINT}',
                    throttle_duration_sec=2.0,
                )
            self._chunk_pub.publish([pose] * self._chunk_len, dt=self._dt, first_index=1, stamp=now.to_msg())
        self._preview_pub.publish(pose_array([pose] * self._chunk_len, frame_id=self._base, stamp=now.to_msg()))

        # The commanded target, stamped with the instant it is due, for the recorder's `action`.
        target_msg = PoseStamped()
        target_msg.header.stamp = (now + Duration(seconds=self._dt)).to_msg()
        target_msg.header.frame_id = self._base
        target_msg.pose = pose
        self._target_pub.publish(target_msg)

        self._stage_pub.publish(Int32(data=self._stage))

        if self._wrist_pub is not None:
            js = JointState()
            js.header.stamp = now.to_msg()
            js.name = WRIST_JOINTS
            js.position = (self._wrist_stiff if self._wrist_is_stiff else self._wrist_soft).tolist()
            self._wrist_pub.publish(js)

        self._tick_count += 1
        if self._tick_count % int(self._hz * 5) == 0:
            self.get_logger().info(
                f'target xyz={np.round(self._target[:3], 3).tolist()} stage={self._stage} '
                f'{"rot" if rotation_mode else "trans"}{"+z" if z_enabled else ""}'
            )

    def destroy_node(self):
        self._input.close()
        self._keys.close()
        return super().destroy_node()


def main() -> None:
    """Entry point registered in setup.py as `spacemouse_teleop`."""
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = SpacemouseTeleop()
    try:
        rclpy.spin(node)          # runs the timer callback until shutdown / Ctrl-C
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C in the terminal, or SIGTERM from `ros2 launch` / `kill`: both are normal exits.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
