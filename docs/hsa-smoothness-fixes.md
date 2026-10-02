# Why the arm is smooth now: the changes between 2026-10-01 and 2026-10-02

The first teleop on the real FER (2026-10-01, bare flange) was slow and jittery: 10 to 20 cm
behind the puck, stalling, stepping. On 2026-10-02 the same teleop tracks within a few
centimetres and moves smoothly. Five changes, in order of effect, with the code behind each.
Measured the same way both days: a 1 kHz recording of the robot's own end-effector pose while
driving (`control_command_success_rate` is the robot's count of torque commands that arrived on
time over the last 100 cycles).

| | 2026-10-01 first teleop | 2026-10-02 |
|---|---|---|
| command success rate (mean / worst 100 ms) | 0.49 | 0.96 / 0.80 |
| tracking gap while driving | 10 to 20 cm, stalls | 1 to 3 cm |
| speed content at the chunk rate, 8 to 12 Hz | stop-and-go | 0% |
| high-frequency speed content, 12 to 40 Hz | | 1% |

## 1. The station's control loop is paced by the robot again (the big one)

`franka_hardware` reads the robot with a call that blocks until the next 1 kHz packet; that read
is what keeps the loop in step with the arm. From Jazzy on, the controller manager re-times its
loop after every late cycle (`overruns.manage`, on by default), which drifts against that read
until half the torque commands miss the robot's window. The robot repeats the previous torque for
every missed cycle, so the arm got a fresh command only every other millisecond, with jitter.
ros2_control issue #2529, filed by Franka Robotics, describes exactly this.

`franka_bringup`'s launch file offers no way to pass the setting, so the station now runs its own
copy of it, `nuc/launch/fer_hardware.launch.py`, identical to the fork's `franka.launch.py` except
that `ros2_control_node` loads `nuc/config/controller_manager.yaml`:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz
    overruns.manage: false
    overruns.print_warnings: false
```

NU-MSR's own `fer_real_controllers.yaml` carries the same two lines. The bringup logs
`Overruns handling is: disabled` and no overrun warnings since.

## 2. Teleop chunks continue along the puck's velocity (the stop-and-go)

The teleop used to send each target twice, as a hold (`[pose] * chunk_len`). The streaming
controller interpolates its reference with a monotone cubic Hermite spline, and a monotone spline
puts a zero tangent at any knot where the path stops, so the reference braked to a halt at every
waypoint and restarted ten times a second. The chunk now carries the target and then the target
continued along the last tick's displacement, which keeps the reference velocity-continuous the
way a policy's 16-step chunk is. `hsa_ros2/teleop_math.py`:

```python
def extrapolate_chunk(target: np.ndarray, prev_target: np.ndarray | None, n: int, box: dict) -> list[np.ndarray]:
    """
    The ``n`` waypoints of a chunk: the target, then the target continued along the step it just took.

    Waypoint ``k`` is the target moved by ``k`` times the last tick's displacement (position
    linearly, orientation by composing the last tick's rotation ``k`` times), clipped to the box.
    A chunk that merely repeats the target as a hold hands the controller's monotone interpolant
    a zero tangent at the waypoint, so the reference decelerates to a stop every tick, ten times a
    second; continuing along the velocity keeps the reference velocity-continuous, as a policy's
    chunk is. With no previous target (a fresh seed) or a zero step the chunk is a plain hold.
    """
    target = np.asarray(target, dtype=np.float64)
    if n <= 1 or prev_target is None:
        return [target.copy() for _ in range(max(1, n))]
    prev = np.asarray(prev_target, dtype=np.float64)
    delta_pos = target[:3] - prev[:3]
    rot_t = st.Rotation.from_rotvec(target[3:])
    delta_rot = rot_t * st.Rotation.from_rotvec(prev[3:]).inv()
    chunk = []
    for k in range(n):
        pose = np.empty(6)
        pose[:3] = target[:3] + k * delta_pos
        pose[3:] = ((delta_rot**k) * rot_t).as_rotvec() if k else target[3:]
        chunk.append(clamp_workspace(pose, box))
    return chunk
```

and in `spacemouse_teleop.py`'s tick, where `prev_target` is the target before this tick's
integration and is `None` after a reseed (so a snapped target is a hold):

```python
        # 4. publish.  Waypoint k of the chunk is due at now + (1 + k) * dt: the first one a full
        #    tick ahead (as the old scripts did), the ones after it continue along the puck's
        #    velocity so the controller's reference never decelerates to a stop between ticks.
        now = self.get_clock().now()
        chunk6 = extrapolate_chunk(self._target, prev_target, self._chunk_len, self._box)
        chunk = [self._pose_msg(p6) for p6 in chunk6]
        pose = chunk[0]
```

`config/teleop.yaml` sets `chunk_len: 3`.

## 3. The elbow holds its posture, but not too hard (nullspace stiffness)

Cartesian teleop holds the flange and leaves the arm's redundant joint free. At serl's default
`nullspace_stiffness: 0.2` the FER's joint 3 drifted to its limit (−2.91 rad against −2.90) within
minutes, which ended the first session with a Desk joint-limit error. The second session ran at
10 and the elbow held, but the arm then stalled 25 cm in front of the ready pose: the target ran
on to the box edge, the arm stayed at x = 0.56 with only 4 N of measured external force, and the
teleop re-seeded the target every time the gap reached 15 cm.

The cause is in the controller's maths. The nullspace torque is `K (q_activation − q)` pushed
through `I − Jᵀ(Jᵀ)⁺`, and that pseudo-inverse is damped (λ = 0.2, serl's `pseudo_inversion.h`),
so the projector is not exact: part of the torque comes out as a Cartesian force pulling the
flange back toward the activation posture, growing with the distance from it. Computed with the
FER's DH model for a flange at z = 0.48 m pointing down, ready-pose posture as the target:

| flange x | leak at K = 10 | at K = 1 |
|---|---|---|
| 0.45 m | 7 N | 0.7 N |
| 0.55 m | 15 N (77% of the 20 N ceiling) | 1.5 N |
| 0.60 m | 25 N (past the ceiling) | 2.6 N |
| 0.70 m | 93 N | 9 N |

The stall at x ≈ 0.56 is where the leak met the clipped spring force. Joint 1's separate stiffness
leaks the same way whenever the arm turns: 100 gave 12 to 49 N at y = 0.2 to 0.3 m, 10 gives 1 to
5 N. `nuc/config/hsa_controllers.yaml` now carries

```yaml
    nullspace_stiffness: 1.0
    joint1_nullspace_stiffness: 10.0
```

five times PolyUMI's elbow hold, under 5 N of leak over the box except its far edge (about 9 N at
x = 0.70), where the arm will still feel heavier. If the elbow wanders again at 1.0, the honest
fix is upstream: the projector's damping as a controller parameter, so it can be small for the
nullspace term where it does not need to protect anything.

## 4. Every controller switch runs inside the realtime loop

Activating an effort controller makes `franka_hardware` switch libfranka into torque control.
Requested from the manager's non-realtime thread, the default, that collides with the read in
flight and libfranka refuses it ("Cannot perform this operation while another control or read
operation is running"); the controller then shows active while the robot never entered torque
mode. The `fer_arm_controller` spawner carries `--switch-asap`, and the servo switch in
`fer_inference.launch.py` and both switches in `fer_home_service.py` set `activate_asap: true`.
NU-MSR's launch carries the same flag.

## 5. The controller activates at all on Jazzy and Kilted

Its `on_activate` handed empty vectors to ros2_control's `get_ordered_interfaces`; Humble grew
them, Jazzy and Kilted throw ("Capacity of ordered_interfaces (0) has to be equal or higher ... for
realtime reasons"). Submodule commit `30bd295` on branch `hsa/jazzy-kilted-activation`:

```diff
@@ -312,6 +312,11 @@ CallbackReturn CartesianImpedanceController::on_activate(
 
   position_interfaces_.clear();
   velocity_interfaces_.clear();
+  // ros2_control from Jazzy on refuses to grow these inside get_ordered_interfaces ("Capacity of
+  // ordered_interfaces (0) has to be equal or higher as size of ordered_names ... for realtime
+  // reasons") and the activation fails; Humble grew them silently. Reserving is a no-op there.
+  position_interfaces_.reserve(joint_names_.size());
+  velocity_interfaces_.reserve(joint_names_.size());
   if (!controller_interface::get_ordered_interfaces(state_interfaces_, joint_names_,
                                                     hardware_interface::HW_IF_POSITION,
                                                     position_interfaces_) ||
```

## The arm keeps going briefly after the puck is let go

Seen on the vertical push on 2026-10-02: the push you hold longest, down to the table, where a
lag is easiest to see. Measured with `ros2_ws/src/hsa_ros2/scripts/spacemouse_tail.py`, which
needs only the puck and reads it two ways at once (every report as it arrives, and one report per
5 ms the way the node's reader thread does):

| | measured, 11 releases |
|---|---|
| reports from the Compact while deflected | one every 8 ms (125/s) |
| reader thread behind the device after a release | 0 to 3 ms |
| the puck's own settle, half-deflection to zero | 50 to 160 ms, 2 to 7 mm at full speed |

So neither the reader nor the puck. The settle is the hand lifting off, and the target follows
it. What remained was the tick: the puck is sampled once per tick, and the first waypoint of every
chunk is one tick ahead (`first_index=1`, as the old scripts' `t_command_target = t_cycle_end + dt`
was), so after the hand is off the target still moves for one to two ticks. At the 10 Hz tick
that is 100 to 200 ms: 2 to 4 cm at full deflection. On top of it the arm trails the reference by
`damping x speed / stiffness`, 89 x 0.2 / 2000 = 9 mm at full speed, which it closes after the
reference stops, in every direction and at any tick rate.

The tick is now 50 Hz (`control_hz` in `config/teleop.yaml`), with `chunk_len: 6` so the chunk
still covers 120 ms if one is late. That leaves 20 to 40 ms of tick, 4 to 8 mm at full deflection,
plus the hand and the trailing distance: about a centimetre at full speed where it was three to
five. Nothing else changes: the controller re-splices every chunk at its first waypoint, the
recorder samples the target at its own rate, and the node holds 50 Hz at 1% of a core (desk check
with the fake puck). The old stack ticked at 10 Hz with the same one-tick lead; it was never
smooth enough for this to be the thing you noticed.

Measured on the arm at 50 Hz (45 s of driving, 17 stops with the arm within 5 cm of the target):
the arm kept moving for 110 to 200 ms after the target stopped and covered a median of 5.7 mm.
The three stops with 21 to 26 mm of follow-through were the ones where the arm was already 27 to
33 mm behind when the target stopped, which is the nullspace stall of section 3, not the tick.

Not the chunk extrapolation: a new chunk's first waypoint is always earlier than the previous
chunk's last, and the controller's `scheduleWaypoint` (UMI's, verbatim) then trims the old
trajectory at the current instant, so the extrapolated waypoints are discarded every tick and
only ever drive the arm if a chunk is late.

## What was not the cause

Raising the force ceiling (`translational_clip` 0.01 to 0.02, 20 to 40 N) made no real
difference and is back at PolyUMI's 20 N. The gains were never the problem; the loop was.

## What remains

After `nuc/station_rt_tune.sh` (performance governor, idle-state hold, applied 2026-10-02) the
success rate's worst 100 ms went from 0.80 to 0.90 at a mean of 0.98. The 100 MB memlock cap is
still in `/etc/security/limits.conf` and takes a re-login to lift; `ros2_control_node` says so at
startup ("Unable to lock the memory") until then.
