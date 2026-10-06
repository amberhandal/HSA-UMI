#!/usr/bin/env python3
"""
FER homing service -- runs ON THE STATION.

Serves /polyumi/home and /polyumi/set_home (std_srvs/Trigger), the same contract and pose file as
nuc/fr3_home_service.py. A home is one FollowJointTrajectory goal straight to fer_arm_controller, timed by
the largest joint delta at `max_joint_speed`, with NO collision check: teach start poses a straight
joint-space interpolation reaches safely, and watch the first home from a new pose with the e-stop in hand.
Joint-space, not Cartesian, so the elbow comes back in the same configuration every trial. Both controllers
claim the same <joint>/effort interfaces, so a home borrows the arm from the servo (STRICT) and hands it back.

    source ~/HSA-UMI/nuc/station_env.sh
    python3 nuc/fer_home_service.py

    ros2 service call /polyumi/set_home std_srvs/srv/Trigger "{}"   # teach (does NOT move)
    ros2 service call /polyumi/home     std_srvs/srv/Trigger "{}"   # MOVES THE ARM
"""

import datetime
import math
import os
import pathlib
import tempfile
import threading

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml

ARM_ID = 'fer'
HOME_JOINT_NAMES = [f'{ARM_ID}_joint{i}' for i in range(1, 8)]
# franka's own `ready` pose.
HOME_JOINTS = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]

# --- The move ---
# Slow on purpose: homing is an operator repositioning the arm between trials, not a policy
# motion, and it runs without a collision check. 0.3 rad/s puts a half-workspace sweep at ~5 s.
DEFAULT_MAX_JOINT_SPEED = 0.3  # rad/s
DEFAULT_MIN_DURATION_S = 3.0
# On top of the move's own duration: a wedge detector, not a deadline.
HOME_EXECUTE_MARGIN_S = 30.0
GOAL_SUBMIT_TIMEOUT_S = 5.0
# How long to let a cancelled goal actually abort before giving up on it. Short: the controller
# stops the trajectory immediately, so this is deceleration, not motion.
CANCEL_TIMEOUT_S = 3.0

# --- Teaching the start pose (the /polyumi/set_home service) ---
# Same file and schema as fr3_home_service.py, so the docs' `delete this file to fall back` holds.
DEFAULT_HOME_POSE_FILE = '~/.ros/polyumi_home_pose.yaml'
HOME_POSE_FILE_VERSION = 1

JOINT_STATE_TOPIC = '/joint_states'
# A taught pose and a home duration are only right if measured from where the arm is *now*; a cached
# joint state approaching a second old means the publisher has stopped.
JOINT_STATE_MAX_AGE_S = 1.0

# --- Controller handover ---
TRAJECTORY_CONTROLLER = 'fer_arm_controller'
SERVO_CONTROLLER = 'polyumi_cartesian_impedance_controller'
SWITCH_TIMEOUT_S = 5.0

ACTION_NAME = f'/{TRAJECTORY_CONTROLLER}/follow_joint_trajectory'


def home_duration_s(current: list[float], target: list[float], max_joint_speed: float, min_duration_s: float) -> float:
    """
    Duration of a home move: the largest joint delta at `max_joint_speed`, never under `min_duration_s`.

    Split out so the one number that sets how fast the arm crosses the workspace is testable
    without a controller on the other end.
    """
    delta = max(abs(c - t) for c, t in zip(current, target))
    return max(min_duration_s, delta / max_joint_speed)


class FerHomeService(Node):
    """Serve /polyumi/home and /polyumi/set_home: drive the FER to a taught start pose, and teach it."""

    def __init__(self, **kwargs):
        """Declare params, load any taught pose, and create the services and controller clients."""
        super().__init__('fer_home_service', **kwargs)

        self.declare_parameter('home_joints', HOME_JOINTS)
        self.declare_parameter('home_pose_file', DEFAULT_HOME_POSE_FILE)
        self.declare_parameter('max_joint_speed', DEFAULT_MAX_JOINT_SPEED)
        self.declare_parameter('min_duration_s', DEFAULT_MIN_DURATION_S)

        self._home_joints = list(self.get_parameter('home_joints').get_parameter_value().double_array_value)
        self._home_source = 'the home_joints parameter'
        self._max_joint_speed = self.get_parameter('max_joint_speed').get_parameter_value().double_value
        self._min_duration_s = self.get_parameter('min_duration_s').get_parameter_value().double_value

        pose_file = self.get_parameter('home_pose_file').get_parameter_value().string_value
        # 'none' (or empty) turns persistence off and homes to the home_joints parameter. A word
        # rather than '' because a shell strips empty quotes off a launch argument.
        self._home_pose_file = (
            pathlib.Path(pose_file).expanduser() if pose_file and pose_file.lower() != 'none' else None
        )
        # A pose taught on the last run outranks the parameter default: it is the more recent and
        # more specific statement of where this task starts.
        taught = self._load_home_pose()
        if taught is not None:
            self._home_joints = taught
            self._home_source = f'the taught pose in {self._home_pose_file}'

        self._cbgroup = ReentrantCallbackGroup()
        self._exec = ActionClient(self, FollowJointTrajectory, ACTION_NAME, callback_group=self._cbgroup)
        self._switch = self.create_client(
            SwitchController, '/controller_manager/switch_controller', callback_group=self._cbgroup
        )
        self.create_service(Trigger, '/polyumi/home', self._on_home, callback_group=self._cbgroup)
        self.create_service(Trigger, '/polyumi/set_home', self._on_set_home, callback_group=self._cbgroup)

        # Our own receive time rather than the message stamp (which a publisher may leave at zero):
        # what matters is how long since we last heard anything.
        self._joint_state_lock = threading.Lock()
        self._joint_state: JointState | None = None
        self._joint_state_at = None
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint_state, 10, callback_group=self._cbgroup)

        # Latest-goal, skip-while-busy: a concurrent /polyumi/home while one is executing is
        # dropped rather than queued.
        self._busy = threading.Lock()

        # Fail loudly at startup rather than on the first /polyumi/home call.
        if not self._exec.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f'{ACTION_NAME} NOT found after 10s -- is bringup.launch.py up with '
                f'{TRAJECTORY_CONTROLLER} spawned?'
            )
        else:
            self.get_logger().info(f'{TRAJECTORY_CONTROLLER} found ({ACTION_NAME} ready).')

        self.get_logger().info(
            'fer_home_service started -- /polyumi/home and /polyumi/set_home are up (std_srvs/Trigger).'
        )
        # Logged loudly, every start: a pose taught weeks ago is indistinguishable from the
        # default once the arm is moving.
        self.get_logger().info(f'Home pose from {self._home_source}: {self._format_joints(self._home_joints)}')

    # ----------------------------------------------------------------------
    # Homing
    # ----------------------------------------------------------------------

    def _on_home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Move the arm to the home joint pose. THIS MOVES THE ARM.

        An explicit, one-shot operator request. Joint-space, straight interpolation from wherever
        the arm stands, at most `max_joint_speed` on the joint that has furthest to go.
        """
        # Read once: /polyumi/set_home rebinds this attribute from another thread.
        home_joints = self._home_joints
        if len(home_joints) != len(HOME_JOINT_NAMES):
            response.success = False
            response.message = f'home_joints has {len(home_joints)} values, expected {len(HOME_JOINT_NAMES)}'
            self.get_logger().error(response.message)
            return response
        if not self._busy.acquire(blocking=False):
            response.success = False
            response.message = 'busy: a home move is already in flight'
            self.get_logger().warning(f'/polyumi/home refused -- {response.message}')
            return response
        try:
            # The duration comes from the delta, and the delta from where the arm is now. Without a
            # fresh joint state the move would run at whatever speed a guessed delta implies.
            current, problem = self._arm_joints_now()
            if current is None:
                response.success = False
                response.message = f'cannot read the arm pose: {problem}'
                self.get_logger().error(f'/polyumi/home refused -- {response.message}')
                return response
            duration_s = home_duration_s(current, home_joints, self._max_joint_speed, self._min_duration_s)

            self.get_logger().warning(
                f'/polyumi/home called -- MOVING THE ARM over {duration_s:.1f}s to the home pose from '
                f'{self._home_source}: {self._format_joints(home_joints)}'
            )
            # Borrow the arm from the streaming controller if it holds it. Unconditional: a switch
            # naming an inactive controller is simply declined, and asking is cheaper than tracking.
            handed_over = self._switch_controllers(activate=TRAJECTORY_CONTROLLER, deactivate=SERVO_CONTROLLER)
            try:
                ok, why = self._run_goal(self._trajectory(home_joints, duration_s), duration_s + HOME_EXECUTE_MARGIN_S)
                if ok:
                    response.success = True
                    response.message = f'homed to the pose from {self._home_source}'
                    self.get_logger().info('Homed.')
                else:
                    response.success = False
                    response.message = f'home move failed: {why}'
                    self.get_logger().error(response.message)
                return response
            finally:
                # Hand the arm back only if we took it. A servo left inactive looks like the policy
                # silently doing nothing, so a failed hand-back fails even a successful home; mutating
                # `response` here reaches the caller, since the return expression was evaluated first.
                if handed_over and not self._switch_controllers(
                    activate=SERVO_CONTROLLER, deactivate=TRAJECTORY_CONTROLLER
                ):
                    response.success = False
                    response.message += (
                        f' -- but failed to hand the arm back to {SERVO_CONTROLLER}; it is still on '
                        f'{TRAJECTORY_CONTROLLER} and the policy cannot drive it. Retry the switch manually.'
                    )
        finally:
            self._busy.release()

    def _trajectory(self, positions: list[float], duration_s: float) -> JointTrajectory:
        """One-point trajectory ending at `positions`, at rest, `duration_s` from now."""
        traj = JointTrajectory()
        traj.joint_names = list(HOME_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        # An explicit zero end velocity makes the controller interpolate a cubic from the current
        # state that comes to rest, rather than a linear ramp that stops dead.
        point.velocities = [0.0] * len(HOME_JOINT_NAMES)
        whole = int(duration_s)
        point.time_from_start = DurationMsg(sec=whole, nanosec=int(round((duration_s - whole) * 1e9)))
        traj.points = [point]
        return traj

    def _run_goal(self, trajectory: JointTrajectory, timeout_s: float) -> tuple[bool, str]:
        """Send the trajectory to fer_arm_controller and block until it finishes; return (ok, why not)."""
        # send_goal_async on a server that is not there hangs until its timeout instead of failing
        # immediately, holding the busy lock past when the caller would otherwise give up.
        if not self._exec.server_is_ready():
            return False, f'{ACTION_NAME} is not available -- is {TRAJECTORY_CONTROLLER} loaded and active?'
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        gf = self._exec.send_goal_async(goal)
        if not self._wait(gf, GOAL_SUBMIT_TIMEOUT_S):
            return False, 'goal submission timed out'
        gh = gf.result()
        if gh is None or not gh.accepted:
            return False, f'goal rejected by {TRAJECTORY_CONTROLLER} (is it ACTIVE? `ros2 control list_controllers`)'
        rf = gh.get_result_async()
        if not self._wait(rf, timeout_s):
            # Cancel and wait for the abort to land: an uncancelled goal keeps driving the arm after
            # the busy lock is released. Bounded, so a server that ignores the cancel cannot wedge us.
            self.get_logger().warning(f'Home move timed out after {timeout_s:.0f}s -- cancelling.')
            gh.cancel_goal_async()
            if not self._wait(rf, CANCEL_TIMEOUT_S):
                self.get_logger().error(
                    'Cancel did not take effect -- the arm may still be moving. Stop it from the Desk UI '
                    'before sending anything else.'
                )
            return False, 'execution timed out'
        res = rf.result()
        if res is None:
            return False, 'no result from the controller'
        if res.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, f'error_code={res.result.error_code} {res.result.error_string}'.rstrip()
        return True, ''

    # ----------------------------------------------------------------------
    # Teaching the start pose
    # ----------------------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        """Cache the latest joint state, with the time we received it."""
        with self._joint_state_lock:
            self._joint_state = msg
            self._joint_state_at = self.get_clock().now()

    def _on_set_home(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Record the arm's current joint positions as the pose /polyumi/home drives to.

        THIS DOES NOT MOVE THE ARM. Put the arm where trials should start -- in Desk's guiding
        mode, with the teleop, or by any other means -- and call this to capture it.
        """
        # Refuse mid-home rather than record a pose the arm is merely passing through.
        if not self._busy.acquire(blocking=False):
            response.success = False
            response.message = 'busy: a home move is in flight -- the arm is moving'
            self.get_logger().warning(f'/polyumi/set_home refused -- {response.message}')
            return response
        try:
            positions, problem = self._arm_joints_now()
            if positions is None:
                response.success = False
                response.message = f'cannot read the arm pose: {problem}'
                self.get_logger().error(f'/polyumi/set_home failed -- {response.message}')
                return response

            self._home_joints = positions
            self._home_source = 'the pose taught over /polyumi/set_home'
            formatted = self._format_joints(positions)

            saved, save_problem = self._save_home_pose(positions)
            if saved:
                self._home_source = f'the taught pose in {self._home_pose_file}'
            response.success = True
            response.message = f'recorded home pose: {formatted}'
            if not saved:
                # Still a success -- the pose is live for this session. Only persistence failed.
                response.message += f' (in memory only -- not saved: {save_problem})'
                self.get_logger().warning(f'/polyumi/set_home could not persist the pose: {save_problem}')
            self.get_logger().info(f'/polyumi/set_home -- {response.message}')
            return response
        finally:
            self._busy.release()

    def _arm_joints_now(self) -> tuple[list[float] | None, str]:
        """Return the 7 arm joint positions from the cached joint state, or (None, why not)."""
        with self._joint_state_lock:
            msg = self._joint_state
            received_at = self._joint_state_at
        if msg is None:
            return None, (
                f'nothing has been published on {JOINT_STATE_TOPIC} -- is bringup.launch.py up with '
                'the joint_state_broadcaster spawned?'
            )
        age_s = (self.get_clock().now() - received_at).nanoseconds / 1e9
        if age_s > JOINT_STATE_MAX_AGE_S:
            return None, (
                f'{JOINT_STATE_TOPIC} is stale ({age_s:.1f}s since the last message, limit '
                f'{JOINT_STATE_MAX_AGE_S:.1f}s) -- the cached positions are not where the arm is now'
            )
        by_name = dict(zip(msg.name, msg.position))
        missing = [name for name in HOME_JOINT_NAMES if name not in by_name]
        if missing:
            return None, f'{JOINT_STATE_TOPIC} carries no {", ".join(missing)}'
        return [float(by_name[name]) for name in HOME_JOINT_NAMES], ''

    def _load_home_pose(self) -> list[float] | None:
        """Return the taught pose from disk, or None if there is none or it is unusable."""
        if self._home_pose_file is None or not self._home_pose_file.is_file():
            return None
        try:
            doc = yaml.safe_load(self._home_pose_file.read_text())
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().error(f'Ignoring {self._home_pose_file}: cannot read it ({exc}).')
            return None
        if not isinstance(doc, dict):
            self.get_logger().error(f'Ignoring {self._home_pose_file}: expected a mapping.')
            return None
        names = doc.get('joint_names')
        positions = doc.get('joint_positions')
        # Names are checked, not just the count: a file written against another arm (fr3_joint*)
        # would otherwise be replayed onto fer_joint1..7 by position and home somewhere else.
        if names != HOME_JOINT_NAMES:
            self.get_logger().error(
                f'Ignoring {self._home_pose_file}: joint_names is {names}, expected {HOME_JOINT_NAMES}.'
            )
            return None
        if not isinstance(positions, list) or len(positions) != len(HOME_JOINT_NAMES):
            self.get_logger().error(
                f'Ignoring {self._home_pose_file}: joint_positions must be {len(HOME_JOINT_NAMES)} values.'
            )
            return None
        try:
            return [float(value) for value in positions]
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f'Ignoring {self._home_pose_file}: joint_positions is not numeric ({exc}).')
            return None

    def _save_home_pose(self, positions: list[float]) -> tuple[bool, str]:
        """Persist the taught pose atomically; return (saved, why not)."""
        if self._home_pose_file is None:
            return False, 'home_pose_file is empty, so persistence is off'
        doc = {
            'version': HOME_POSE_FILE_VERSION,
            'joint_names': HOME_JOINT_NAMES,
            'joint_positions': positions,
            'recorded_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        }
        header = (
            '# HSA-UMI eval start pose, written by /polyumi/set_home (nuc/fer_home_service.py).\n'
            '# /polyumi/home drives the arm here. Delete this file to fall back to the\n'
            "# home_joints parameter (franka's `ready` pose by default).\n"
        )
        try:
            self._home_pose_file.parent.mkdir(parents=True, exist_ok=True)
            # Write-and-rename: a crash partway through a plain write leaves a truncated file that
            # the next startup refuses to parse, silently homing to the default instead.
            fd, tmp_path = tempfile.mkstemp(dir=str(self._home_pose_file.parent), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as handle:
                    handle.write(header)
                    yaml.safe_dump(doc, handle, default_flow_style=False, sort_keys=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self._home_pose_file)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            return False, f'{self._home_pose_file}: {exc}'
        return True, ''

    @staticmethod
    def _format_joints(positions) -> str:
        """Render joint positions for a log line or a service response."""
        return '[' + ', '.join(f'{value:.4f}' for value in positions) + '] rad'

    def _switch_controllers(self, *, activate: str, deactivate: str) -> bool:
        """
        Swap which controller drives the arm; return whether the switch actually happened.

        STRICT, not BEST_EFFORT: a partial switch would leave the arm with either two controllers
        claiming its effort interfaces or none at all. A False return is not always an error --
        asking to deactivate a controller that was never spawned fails the same way, which is the
        normal case for a bare bringup without the streaming controller.
        """
        if not self._switch.wait_for_service(timeout_sec=SWITCH_TIMEOUT_S):
            self.get_logger().warning(
                'controller_manager switch_controller not available; leaving controllers as they are.'
            )
            return False
        request = SwitchController.Request()
        request.activate_controllers = [activate]
        request.deactivate_controllers = [deactivate]
        request.strictness = SwitchController.Request.STRICT
        # Switch inside the realtime loop, between read() and write(): from the non-realtime thread
        # libfranka refuses the effort-mode change ("another control or read operation is running").
        request.activate_asap = True
        future = self._switch.call_async(request)
        if not self._wait(future, SWITCH_TIMEOUT_S):
            self.get_logger().error(f'switch to {activate} timed out after {SWITCH_TIMEOUT_S}s')
            return False
        ok = bool(future.result() and future.result().ok)
        if ok:
            self.get_logger().info(f'Controller switched: {deactivate} -> {activate}')
        else:
            self.get_logger().info(
                f'Controller switch {deactivate} -> {activate} declined; '
                f'{deactivate} is probably not running. Continuing.'
            )
        return ok

    @staticmethod
    def _wait(future, timeout_s: float) -> bool:
        """Block until future completes or times out (the node spins under a MultiThreadedExecutor)."""
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        return done.wait(timeout=timeout_s)


def main():
    """Spin the service node under a multithreaded executor."""
    rclpy.init()
    node = FerHomeService()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
