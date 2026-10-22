"""
SpaceMouse teleoperation of the Franka TCP through the streaming Cartesian impedance controller.

Integrates a target pose and publishes it as a small action chunk on the controller's topic, the same
message the policy client publishes at inference, so teleop and policy drive the arm through one path.
Topic and service names are parameters (config/teleop.yaml). Run it in its OWN terminal (keyboard input
needs a TTY):

    ros2 run hsa_ros2 spacemouse_teleop --ros-args --params-file $(ros2 pkg prefix --share hsa_ros2)/config/teleop.yaml
    ros2 run hsa_ros2 spacemouse_teleop --ros-args -p execute:=true ...       # MOVES THE ARM
    ros2 run hsa_ros2 spacemouse_teleop --ros-args -p input_source:=fake_sine # desk test, no device

Keys: c start episode, s save, f save as failure, Backspace drop, 1..N set the subtask (stage_names), Space next
subtask, n skip to the next plan row,
p print pose, r re-seed target from TF, x toggle wrist stiff/soft, q quit.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from controller_manager_msgs.msg import ControllerManagerActivity
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from lifecycle_msgs.msg import State
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from tf2_ros import ConnectivityException, ExtrapolationException, LookupException  # type: ignore[attr-defined]

# The controller's chunk topic and preview format. Importing them from polyumi_ros2 keeps the topic
# name defined in exactly one place.
from polyumi_ros2.target_chunk import CONSUMER_HINT, TargetChunkPublisher, pose_array

from hsa_ros2.dataset_writer import DEFAULT_STAGE_NAMES
from hsa_ros2.teleop_math import (
    extrapolate_chunk,
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
    Read the SpaceMouse in a background thread and hand out the latest state.

    pyspacemouse's ``read()`` consumes ONE USB report per call and the device sends them far faster than
    the control rate; reading only in the control loop would build a backlog of stale input, so a thread
    drains the device continuously and the control loop just takes a snapshot.
    """

    def __init__(self, deadzone: float, operator_facing: str = 'legacy'):
        """Open the device and start the reader thread."""
        import pyspacemouse  # imported here so the fake/none sources work on machines without it

        self._device = pyspacemouse.open()
        if not self._device:
            raise RuntimeError(
                'pyspacemouse.open() failed: is the SpaceMouse plugged in and the hidraw udev rule installed?'
            )
        self._deadzone = deadzone
        self._facing = operator_facing
        self._lock = threading.Lock()
        self._axes = np.zeros(6)
        self._buttons = [False, False]
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name='spacemouse_reader', daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # The Compact sends a report every 8 ms while deflected (125/s, measured with
        # scripts/spacemouse_tail.py); a read every 5 ms keeps the snapshot within 3 ms of the newest.
        while not self._stop:
            s = self._device.read()
            raw = [s.x, s.y, s.z, s.roll, s.pitch, s.yaw]
            buttons = list(getattr(s, 'buttons', []) or [])
            with self._lock:
                self._axes = map_spacemouse_axes(raw, self._deadzone, self._facing)
                self._buttons = [bool(buttons[i]) if i < len(buttons) else False for i in range(2)]
            time.sleep(0.005)

    def read(self):
        """Return a snapshot: (axes[6] in [-1, 1], buttons[2])."""
        with self._lock:
            return self._axes.copy(), list(self._buttons)

    def close(self) -> None:
        """Stop the reader thread and close the device."""
        self._stop = True
        try:
            self._device.close()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass


class FakeSineSource:
    """Desk-test stand-in: a slow circle, no buttons.  See teleop_math.fake_sine_axes."""

    def __init__(self, period_s: float, amplitude: float):
        """Start the clock the sine runs on."""
        self._t0 = time.monotonic()
        self._period = period_s
        self._amp = amplitude

    def read(self):
        """Return the sine axes for the current time; the z button reads as held."""
        return fake_sine_axes(time.monotonic() - self._t0, self._period, self._amp), [False, True]

    def close(self) -> None:
        """Do nothing: there is no device."""
        pass


class NoInputSource:
    """Keyboard-only session: the target never moves, but every topic/service hook still works."""

    def read(self):
        """Return zero axes and no buttons."""
        return np.zeros(6), [False, False]

    def close(self) -> None:
        """Do nothing: there is no device."""
        pass


# ----------------------------------------------------------------------------------------------
# Keyboard: one character at a time from the terminal, without waiting for Enter.
# ----------------------------------------------------------------------------------------------


class KeyReader:
    """
    Put every keypress on a queue from a background thread.

    Uses the terminal's *cbreak* mode (characters arrive immediately) and restores the terminal on
    close.  If stdin is not a terminal (e.g. started by `ros2 launch`) the reader disables itself and
    says so, rather than crashing.
    """

    def __init__(self, logger):
        """Switch the terminal to cbreak mode and start the reader thread, if stdin is a TTY."""
        self.keys: queue.Queue[str] = queue.Queue()
        self._enabled = sys.stdin.isatty()
        self._old_settings = None
        if not self._enabled:
            logger.warning(
                'stdin is not a TTY: keyboard commands disabled (run the node with `ros2 run` in a terminal).'
            )
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
        """Stop the reader and restore the terminal settings."""
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
        """Declare parameters, open the input, and wire up TF, publishers, recorder clients and the timer."""
        super().__init__('spacemouse_teleop')

        # --- Parameters (defaults; override with --params-file, -p name:=value or a launch file) ---
        self.declare_parameter('control_hz', 10.0)
        self.declare_parameter('base_frame', 'fer_link0')
        self.declare_parameter('tcp_frame', 'hsa_tcp')
        # false = publish the preview only (nothing moves).  true = publish chunks to the
        # controller topic.  Deliberately not toggleable from the keyboard.
        self.declare_parameter('execute', False)
        # Waypoints per chunk: the target, then more along the puck's velocity, so a late tick leaves
        # the controller a reference to follow instead of running out of trajectory.
        self.declare_parameter('chunk_len', 3)
        # Beyond this distance the target snaps back onto the arm (home move, e-stop, reach limit, hand on
        # the arm, controller restart); otherwise a stale target becomes a lunge when it is followed again.
        self.declare_parameter('reseed_distance', 0.15)
        self.declare_parameter('max_pos_speed', 0.20)  # m/s at full deflection
        self.declare_parameter('max_rot_speed', 0.15)  # rad/s at full deflection
        # Workspace box in the base frame.
        self.declare_parameter('workspace.x_min', 0.2)
        self.declare_parameter('workspace.x_max', 0.9)
        self.declare_parameter('workspace.y_min', -0.4)
        self.declare_parameter('workspace.y_max', 0.4)
        self.declare_parameter('workspace.z_min', 0.05)
        self.declare_parameter('workspace.z_max', 0.5)
        # 'spacemouse' | 'fake_sine' | 'none'
        self.declare_parameter('input_source', 'spacemouse')
        self.declare_parameter('spacemouse.deadzone', 0.0)
        # Which robot axis the operator looks along (+x, -x, +y, -y): pushes move the TCP the way the hand goes.
        # 'legacy' keeps the original UMI permutation, which swaps forward/back with up/down on pyspacemouse.
        self.declare_parameter('spacemouse.operator_facing', 'legacy')
        # base: buttons switch translate/rotate (rotation about base axes). tool_twist: x, y, z and a knob twist
        # together, the twist spinning the tool about its own axis; the rotation button still gives 'base'.
        self.declare_parameter('spacemouse.rotation_style', 'base')
        self.declare_parameter('spacemouse.twist_sign', 1.0)
        # false: up/down works without holding the z button.
        self.declare_parameter('spacemouse.z_needs_button', True)
        self.declare_parameter('spacemouse.rotation_button', 0)  # held -> rotate, translation locked
        self.declare_parameter('spacemouse.z_button', 1)  # held -> vertical motion allowed
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
        # Stage keys pressed in collect_ui's window arrive here; this node stays the only stage publisher.
        self.declare_parameter('stage_request_topic', '/hsa/stage_request')
        # Subtask names by stage index; keys 1..N select them, c resets to the first.
        self.declare_parameter('stage_names', list(DEFAULT_STAGE_NAMES))
        self.declare_parameter('wrist_target_topic', '/hsa/target_servo')
        self.declare_parameter('preview_topic', '/polyumi/target_poses_preview')
        # The controller that executes the chunks, and the controller manager's (latched) state-change topic.
        # A home switches it off; the subscriber count on the chunk topic alone can miss that, and a target
        # left where the operator was is then followed again the moment the home hands the arm back.
        self.declare_parameter('servo_controller', 'polyumi_cartesian_impedance_controller')
        self.declare_parameter('controller_activity_topic', '/controller_manager/activity')

        p = lambda name: self.get_parameter(name).value  # noqa: E731 - short accessor
        self._hz = float(p('control_hz'))
        self._dt = 1.0 / self._hz
        self._base = str(p('base_frame'))
        self._tcp = str(p('tcp_frame'))
        self._execute = bool(p('execute'))
        self._chunk_len = max(1, int(p('chunk_len')))
        self._reseed_distance = float(self.get_parameter('reseed_distance').value)
        self._max_pos_speed = float(p('max_pos_speed'))
        self._max_rot_speed = float(p('max_rot_speed'))
        self._box = {k: float(p(f'workspace.{k}')) for k in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')}
        self._rotation_button = int(p('spacemouse.rotation_button'))
        self._z_button = int(p('spacemouse.z_button'))
        self._rotation_style = str(p('spacemouse.rotation_style'))
        self._twist_sign = float(p('spacemouse.twist_sign'))
        self._z_needs_button = bool(p('spacemouse.z_needs_button'))
        self._wrist_enabled = bool(p('wrist.enabled'))
        self._wrist_soft = np.asarray(p('wrist.setpoint_rad'), dtype=np.float64)
        self._wrist_stiff = np.asarray(p('wrist.stiff_setpoint_rad'), dtype=np.float64)
        self._wrist_is_stiff = False

        # --- Input ------------------------------------------------------------------------
        source = str(p('input_source'))
        if source == 'spacemouse':
            self._input = SpacemouseSource(float(p('spacemouse.deadzone')), str(p('spacemouse.operator_facing')))
        elif source == 'fake_sine':
            self._input = FakeSineSource(float(p('fake_sine.period_s')), float(p('fake_sine.amplitude')))
        elif source == 'none':
            self._input = NoInputSource()
        else:
            raise ValueError(f"input_source must be spacemouse | fake_sine | none, got '{source}'")
        self._keys = KeyReader(self.get_logger())
        # True while executing with no subscriber on the chunk topic, or the servo controller not active (see _tick).
        self._holding = False
        # Last reported state of servo_controller: None until the controller manager has said anything.
        self._servo_active: bool | None = None
        self._servo_controller = str(p('servo_controller'))

        # --- TF: where is the TCP right now? ----------------------------------------------
        # Only the latest base->tcp transform is needed: it seeds the target where the arm actually is.
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._target: np.ndarray | None = None  # [x, y, z, rx, ry, rz], base frame

        # --- Publishers -------------------------------------------------------------------
        # Publishing where nobody listens is legal and silent, hence the subscriber check in _tick.
        self._chunk_pub = None
        if self._execute:
            self._chunk_pub = TargetChunkPublisher(self, frame_id=self._base, joint_name=self._tcp)
        self._preview_pub = self.create_publisher(PoseArray, str(p('preview_topic')), 10)
        self._target_pub = self.create_publisher(PoseStamped, str(p('teleop_target_topic')), 10)
        self._stage_pub = self.create_publisher(Int32, str(p('stage_topic')), 10)
        self._wrist_pub = (
            self.create_publisher(JointState, str(p('wrist_target_topic')), 10) if self._wrist_enabled else None
        )
        self._stage = 0
        self._stage_names = [str(n) for n in p('stage_names')]
        self.create_subscription(Int32, str(p('stage_request_topic')), self._stage_request_cb, 10)
        if self._execute:
            latched = QoSProfile(
                depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            self.create_subscription(
                ControllerManagerActivity, str(p('controller_activity_topic')), self._activity_cb, latched
            )

        # --- Service clients on the recorder ----------------------------------------------
        # Calls are asynchronous (call_async + done callback) so the control loop never blocks on the recorder.
        prefix = str(p('recorder_service_prefix'))
        self._recorder = {
            name: self.create_client(Trigger, f'{prefix}/{name}')
            for name in ('start_episode', 'stop_episode', 'stop_failure', 'drop_episode', 'next')
        }

        # --- Control timer: rclpy calls _tick every 1/hz seconds from spin() ----------------
        self._tick_count = 0
        self.create_timer(self._dt, self._tick)

        mode = (
            f'EXECUTE via {self._chunk_pub.topic_name} (ARM WILL MOVE)'
            if self._execute
            else 'preview only (nothing moves)'
        )
        self.get_logger().info(
            f'spacemouse_teleop: {mode} | input={source} | {self._hz:g} Hz | '
            f'{self._max_pos_speed} m/s, {self._max_rot_speed} rad/s | frames {self._base} -> {self._tcp} | '
            f'rotation {self._rotation_style}, z {"button" if self._z_needs_button else "free"}, '
            f'operator facing {self.get_parameter("spacemouse.operator_facing").value}'
        )
        self.get_logger().info(
            f'keys: c start  s save  f save as failure  Backspace drop  1-{len(self._stage_names)} subtask '
            f'({", ".join(self._stage_names)})  Space next subtask  n skip plan row  '
            'p pose  r reseed  x wrist  q quit'
        )

    # ------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------

    def _lookup_tcp(self) -> np.ndarray | None:
        """Return the latest base->tcp transform as a 6-vector, or None with a throttled warning if TF is not up."""
        try:
            tf = self._tf_buffer.lookup_transform(self._base, self._tcp, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warning(
                f'TF {self._base} -> {self._tcp} not available yet ({type(e).__name__}); '
                'is the arm bringup running and publishing hsa_tcp?',
                throttle_duration_sec=2.0,
            )
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return np.concatenate([[t.x, t.y, t.z], quat_to_rotvec(r.x, r.y, r.z, r.w)])

    def _pose_msg(self, pose6: np.ndarray) -> Pose:
        """Convert a 6-vector to a geometry_msgs/Pose (position + quaternion)."""
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
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'{name} failed: {e!r}')
                return
            # Two separate call sites: rclpy refuses to log from one line at a changing severity.
            if resp.success:
                self.get_logger().info(f'{name}: {resp.message}')
            else:
                self.get_logger().error(f'{name}: {resp.message}')

        future.add_done_callback(done)

    def _activity_cb(self, msg: ControllerManagerActivity) -> None:
        """Track whether servo_controller is active; a controller missing from the list is not."""
        states = {c.name: c.state.id for c in msg.controllers}
        active = states.get(self._servo_controller) == State.PRIMARY_STATE_ACTIVE
        if active != self._servo_active:
            self.get_logger().info(f'{self._servo_controller}: {"active" if active else "not active"}')
        self._servo_active = active

    def _stage_request_cb(self, msg: Int32) -> None:
        """Apply a stage key pressed in collect_ui, as if it had been pressed here."""
        if self._stage != int(msg.data):
            self._stage = int(msg.data)
            self.get_logger().info(f'stage -> {self._stage_label()} (from collect_ui)')

    def _stage_label(self) -> str:
        """Return 'index (name)' for the current stage, or just the index past the named ones."""
        names = self._stage_names
        return f'{self._stage} ({names[self._stage]})' if self._stage < len(names) else str(self._stage)

    def _handle_key(self, ch: str) -> bool:
        """Dispatch one keypress.  Returns False when the node should quit."""
        if ch == 'q':
            return False
        if ch == 'c':
            self._stage = 0
            self._call_recorder('start_episode')
            self.get_logger().info(f'stage -> {self._stage_label()}')
        elif ch == 's':
            self._call_recorder('stop_episode')
        elif ch == 'f':
            self._call_recorder('stop_failure')
        elif ch in ('\x7f', '\x08'):  # Backspace
            self._call_recorder('drop_episode')
        elif ch == ' ':
            self._stage += 1
            self.get_logger().info(f'stage -> {self._stage_label()}')
        elif ch.isdigit() and 1 <= int(ch) <= len(self._stage_names):
            self._stage = int(ch) - 1
            self.get_logger().info(f'stage -> {self._stage_label()}')
        elif ch == 'n':
            self._call_recorder('next')
        elif ch == 'p':
            meas = self._lookup_tcp()
            self.get_logger().info(f'target {np.round(self._target, 4).tolist() if self._target is not None else None}')
            self.get_logger().info(f'measured {np.round(meas, 4).tolist() if meas is not None else None}')
        elif ch == 'r':
            self._target = None  # next tick re-seeds from TF
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
            self.get_logger().info(
                f'target seeded from TF: {np.round(self._target, 4).tolist()}', throttle_duration_sec=5.0
            )

        # 3. read the input and advance the target by one tick
        prev_target = self._target.copy()
        axes, buttons = self._input.read()
        rotation_mode = bool(buttons[self._rotation_button]) if self._rotation_button < len(buttons) else False
        z_enabled = (not self._z_needs_button) or (
            bool(buttons[self._z_button]) if self._z_button < len(buttons) else False
        )
        self._target = clamp_workspace(
            integrate_target(
                self._target,
                axes,
                max_pos_speed=self._max_pos_speed,
                max_rot_speed=self._max_rot_speed,
                hz=self._hz,
                rotation_mode=rotation_mode,
                z_enabled=z_enabled,
                rotation_style=self._rotation_style,
                twist_sign=self._twist_sign,
            ),
            self._box,
        )

        # 3b. an arm far from the target is not following it: snap the target onto the arm and start there
        measured = self._lookup_tcp()
        if measured is not None and float(np.linalg.norm(measured[:3] - self._target[:3])) > self._reseed_distance:
            self.get_logger().warning(
                f'arm is {float(np.linalg.norm(measured[:3] - self._target[:3])):.2f} m from the target: '
                'reseeding the target on the arm',
                throttle_duration_sec=2.0,
            )
            self._target = clamp_workspace(measured, self._box)
            prev_target = None  # a snapped target has no velocity: the chunk is a hold

        # 4. publish. Waypoint k is due at now + (1 + k) * dt; the ones after the first continue along the
        #    puck's velocity so the controller's reference never decelerates to a stop between ticks.
        now = self.get_clock().now()
        chunk6 = extrapolate_chunk(self._target, prev_target, self._chunk_len, self._box)
        chunk = [self._pose_msg(p6) for p6 in chunk6]
        pose = chunk[0]
        if self._chunk_pub is not None:
            if self._chunk_pub.get_subscription_count() == 0 or self._servo_active is False:
                # Hold: with nobody executing the chunks the target would drift from the arm and the arm
                # would jump when the controller activates. Re-seeding every tick keeps the target on the arm.
                if not self._holding:
                    why = (
                        f'{self._servo_controller} is not active (a home, or a controller switch)'
                        if self._servo_active is False
                        else f'nothing is subscribed to {self._chunk_pub.topic_name}'
                    )
                    self.get_logger().warning(
                        f"{why}: holding the target at the arm's pose, puck input ignored until the controller "
                        f'subscribes. Needs {CONSUMER_HINT}'
                    )
                    self._holding = True
                self._target = None
            elif self._holding:
                self._holding = False
                self.get_logger().info("controller subscribed: following the puck from the arm's current pose")
            self._chunk_pub.publish(chunk, dt=self._dt, first_index=1, stamp=now.to_msg())
        self._preview_pub.publish(pose_array(chunk, frame_id=self._base, stamp=now.to_msg()))

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
            # From the published pose, not self._target: that is None while holding (e.g. during a home).
            xyz = [round(pose.position.x, 3), round(pose.position.y, 3), round(pose.position.z, 3)]
            self.get_logger().info(
                f'target xyz={xyz} stage={self._stage} '
                f'{"rot" if rotation_mode else "trans"}{"+z" if z_enabled else ""}'
                f'{" (holding: no controller)" if self._holding else ""}'
            )

    def destroy_node(self):
        """Close the input and the keyboard reader, then destroy the node."""
        self._input.close()
        self._keys.close()
        return super().destroy_node()


def main() -> None:
    """Entry point registered in setup.py as `spacemouse_teleop`."""
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = SpacemouseTeleop()
    try:
        rclpy.spin(node)  # runs the timer callback until shutdown / Ctrl-C
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C in the terminal, or SIGTERM from `ros2 launch` / `kill`: both are normal exits.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
