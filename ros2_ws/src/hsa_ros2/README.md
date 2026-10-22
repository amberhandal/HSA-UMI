# hsa_ros2 — SpaceMouse teleop and demo recording for the HSA wrist on the Franka FER

Drives the FER by SpaceMouse through the streaming impedance controller, records demonstrations
straight from the robot into the UMI ReplayBuffer format the diffusion-policy fork trains on, and
launches the camera plus `policy_client_node` for deployment. It sits *next to* `polyumi_ros2`,
imports its wiring (`polyumi_ros2.target_chunk`, `polyumi_ros2.camera_preproc`) and never modifies
it. Teleop and policy drive the arm through one identical path: the action chunk on
`/polyumi/target_poses_traj`.

```
 SpaceMouse ──USB──▶ spacemouse_teleop ──/polyumi/target_poses_traj──▶ streaming impedance controller ──▶ FER
      keyboard ──────▶      │                  (MultiDOFJointTrajectory)        (ros2_control plugin, 1 kHz)
                            ├──/polyumi/target_poses_preview──▶ Foxglove / RViz (nothing moves)
                            ├──/hsa/teleop_target ───────────▶ demo_recorder  (becomes the `action` key)
                            ├──/hsa/stage ───────────────────▶ demo_recorder
                            ├──/hsa/target_servo ────────────▶ wrist command node ──▶ Dynamixel driver       [next]
                            └──services /hsa_recorder/{start,stop,drop}_episode, next, export_zip ──▶ demo_recorder
 TF: fer_link0 → hsa_tcp (from the arm bringup) ──▶ spacemouse_teleop (seeds the target), demo_recorder
```

## Contents

| File | Purpose |
|---|---|
| `hsa_ros2/spacemouse_teleop.py` | The teleop node: a 200 Hz thread drains the SpaceMouse; a timer integrates the target and publishes an action chunk (`execute: true`) and a preview; keyboard keys call the recorder's services. Read its module docstring, then `_tick()`. |
| `hsa_ros2/teleop_math.py` | ROS-free arithmetic: axis mapping, per-tick integration with the button modes, workspace clamp, quaternion conversions, the fake-sine input. |
| `hsa_ros2/demo_recorder.py` | The recorder node: one stored step per camera frame, every stream sampled at that frame's capture instant; serves the five Trigger services. |
| `hsa_ros2/dataset_writer.py` | ROS-free: the UMI ReplayBuffer layout (key names, dtypes, Blosc chunks), episode assembly, the append-only zarr store, zip export, time-alignment helpers. |
| `hsa_ros2/export_dataset.py` | ROS-free CLI: summarise a recorded dataset, list the per-episode sidecars, pack it into `.zarr.zip`, or reopen a zip the way `UmiDataset` does. |
| `hsa_ros2/fake_camera.py` | Synthetic rgb8 Image publisher for desk tests (same type, encoding and rate as a RealSense colour stream). |
| `hsa_ros2/diag_stats.py` | Collects `policy_client_node`'s `/polyumi/diag/*` scalars for N seconds and prints mean / p50 / p95 per topic. |
| `config/teleop.yaml`, `config/recorder.yaml`, `config/inference.yaml` | Commented parameters for the teleop node, the recorder and `policy_client_node`. `execute` and `execute_motion` default to false. |
| `launch/teleop.launch.xml` | The teleop node with `config/teleop.yaml`; exposes `execute` and `input_source`. |
| `launch/camera.launch.xml` | The RealSense colour stream on `/camera/color/image_raw`, 640×480 at 30 Hz (4:3, so the shared crop is a no-op), depth and IMU off. |
| `launch/inference.launch.xml` | Deployment on the laptop: the camera plus `policy_client_node` with `config/inference.yaml`, optional Foxglove. |
| `launch/desk_test.launch.xml` | Fake arm TF + fake camera + teleop on synthetic input + recorder, no hardware. |
| `launch/rviz.launch.xml` | RViz with the arm, TF, the commanded poses, the wrench and the camera (`rviz/hsa.rviz`). |
| `scripts/desk_test.sh` | Runs the desk-test launch, drives the recorder through its services and prints the dataset summary. |
| `scripts/validate_umi_dataset.py` | Loads a `.zarr.zip` with UMI's real `UmiDataset` (analysis venv) and checks the (16, 10) action shape. |
| `scripts/bench_policy_inference.py` | Times `predict_action` on this machine's GPU (analysis venv). |
| `scripts/make_dummy_checkpoint.py` | Runs inside the `polyumi-dp` image: writes a random-weights checkpoint in the fork's own format and reloads it through `serve_policy`'s loader. |
| `scripts/loopback_inference_test.sh` | Serves a checkpoint from the container on `:8002` and runs the real `policy_client_node` against it with a fake camera and a fake arm transform; prints the round-trip statistics. |
| `scripts/station_mock_test.sh` | The deployment topology with the arm off: mock bringup on the station over ssh, camera, container and client here, TF over the cable. |
| `scripts/spacemouse_tail.py` | Tells the puck's own settle tail from the reader's backlog (no ROS, no arm). |
| `test/test_teleop_math.py`, `test/test_dataset_writer.py` | pytest, no ROS needed. |

## Topics, services and frames

| Name | Type | Producer → consumer |
|---|---|---|
| `/polyumi/target_poses_traj` | `trajectory_msgs/MultiDOFJointTrajectory` | teleop (`execute: true`) or `policy_client_node` (`execute_motion: true`) → the streaming impedance controller on the station. Publishing with no subscriber is legal and silent, so the teleop warns when nothing listens. |
| `/polyumi/target_poses_preview` | `geometry_msgs/PoseArray` | teleop and `policy_client_node` → Foxglove / RViz; nothing moves |
| `/hsa/teleop_target` | `geometry_msgs/PoseStamped` | teleop → recorder; stored as the `action` key |
| `/hsa/stage` | `std_msgs/Int32` | teleop (Space increments it) → recorder |
| `/hsa/wrist/state` | `hsa_wrist_msgs/WristState` | `wrist_state_publisher` (wrist_control workspace) → recorder (`wrist.enabled: true`): extension and command angles, ODrive position/velocity/torque/Iq, bus V/I, temperatures, errors, per axis |
| `/platform_pose`, `/platform_tag_visible` | `geometry_msgs/PoseStamped`, `std_msgs/Bool` | camera_tracking `platform_pose` → recorder (`plate.enabled: true`): the top plate in its rest frame |
| `/tracking_cam/image_raw` (or `/compressed`) | `sensor_msgs/Image` 1280×800 | baseplate camera → recorder (`wrist_camera.enabled: true`), stored as `camera1_rgb` 224×224 |
| `/camera/color/image_raw` | `sensor_msgs/Image`, rgb8, 640×480 @ 30 Hz | `camera.launch.xml` or `fake_camera` → recorder and `policy_client_node` |
| `/franka_robot_state_broadcaster/external_wrench_in_base_frame` | `geometry_msgs/WrenchStamped` | station → recorder (`wrench.enabled: true`) |
| `/polyumi/diag/*` | `std_msgs/Float32` | `policy_client_node` → `diag_stats` |
| `/hsa_recorder/{start_episode,stop_episode,drop_episode,next,export_zip}` | `std_srvs/Trigger` | teleop keys `c` / `s` / Backspace / `n` (or `ros2 service call`) → recorder |
| TF `fer_link0 → hsa_tcp` | | arm bringup (`nuc/launch/bringup.launch.py`; a static transform in desk tests) → teleop (seeds the target) and recorder (the eef pose) |

`hsa_tcp` is the HSA wrist plate at the neutral servo setpoint, a static child of the flange
(`fer_link8`) and the frame the streaming controller commands. Every node takes the frame name as a
parameter (`tcp_frame` / `eef_frame`); the dataset keys (`robot0_eef_pos` and friends) are UMI's
and stay as they are.

| Service (`std_srvs/Trigger`) | Effect |
|---|---|
| `/hsa_recorder/start_episode` | begin buffering steps (refused if no frame or no TF yet) |
| `/hsa_recorder/stop_episode` | save the episode; warns if shorter than `min_steps` |
| `/hsa_recorder/drop_episode` | discard the episode in progress, or the last saved one if idle |
| `/hsa_recorder/next` | skip to the next plan row (rows also advance automatically when an episode is saved) |
| `/hsa_recorder/export_zip` | write `<output_dir>/<name>.zarr.zip` |

## Build

On the machine that runs it (the station under Humble, or this PC under Kilted for desk tests):

```bash
cd ~/hsa-ws/HSA-UMI
git submodule update --init external/franka_streaming_impedance_controller   # once
cd ros2_ws
bash -c 'unset VIRTUAL_ENV; source /opt/ros/<distro>/setup.bash && colcon build --symlink-install --packages-select franka_streaming_impedance_client polyumi_ros2 hsa_ros2'
source install/setup.bash
```

`--symlink-install` means edits to the Python files take effect without rebuilding (rebuild after
adding files or changing `setup.py`). `polyumi_ros2` declares a dependency on `polyumi_pi_msgs`
(the Raspberry Pi's protobuf messages), whose build needs a `grpc_tools` that can generate `.pyi`
stubs; the Pi is never used here, so either skip it with `--packages-ignore polyumi_pi_msgs` (what
the command above amounts to on a machine without it) or `pip install --user --break-system-packages grpcio-tools`
once and build it.

Python dependencies under ROS's interpreter, plus the SpaceMouse udev rule:

```bash
pip install --user --break-system-packages pyspacemouse easyhid
pip install --user --break-system-packages 'zarr==2.18.4' 'numcodecs<0.14'
sudo sh -c 'echo "KERNEL==\"hidraw*\", SUBSYSTEM==\"hidraw\", MODE=\"0664\", GROUP=\"plugdev\"" > /etc/udev/rules.d/99-hidraw-permissions.rules' && sudo usermod -aG plugdev $USER
```

Two gotchas for anyone editing the package:

- An XML comment in `package.xml` may not contain `--` (for example a pip flag). If it does,
  `catkin_pkg` cannot parse the manifest, colcon quietly builds the package as plain Python without
  ROS hooks, `colcon build` still says "Finished", and `ros2 run` answers "Package not found".
  `cat install/<pkg>/share/colcon-core/packages/<pkg>` shows the symptom: it lists `setuptools`
  instead of the ROS dependencies.
- Use `get_logger().warning(...)`, never the `warn(...)` alias, with `throttle_duration_sec=` or
  `once=`: rclpy identifies a log call site by its stack frame, the deprecated alias wraps every
  call in the same frame, and the second call with different filters raises
  `ValueError: Requested logging filters cannot be changed between calls`.

## Test

Unit tests, no ROS needed: `cd ros2_ws/src/hsa_ros2 && PYTHONPATH=. pytest test/ -q`.

The whole collection loop with no hardware (a static transform for the arm, `fake_camera`, the
teleop on `fake_sine`, the recorder), then the proof that the file is trainable with UMI's own
dataset class (analysis venv, not ROS):

```bash
cd ~/hsa-ws/HSA-UMI/ros2_ws/src/hsa_ros2 && scripts/desk_test.sh /tmp/hsa_desk
~/hsa-ws/diffusion_policy/venv/bin/python scripts/validate_umi_dataset.py /tmp/hsa_desk/hsa_desk.zarr.zip
```

Expected: `saved episode 0: ~150 steps`, `saved episode 1`, `exported ...zarr.zip`, a summary
listing nine `data/` keys, zero tracebacks, and finally `action: (16, 10) ... VALIDATION OK`.

The teleop node alone, no robot, in three terminals:

```bash
source /opt/ros/<distro>/setup.bash && ros2 run tf2_ros static_transform_publisher --x 0.5 --y 0.0 --z 0.3 --roll 3.14159 --pitch 0 --yaw 0 --frame-id fer_link0 --child-frame-id hsa_tcp
source ~/hsa-ws/HSA-UMI/ros2_ws/install/setup.bash && ros2 run hsa_ros2 spacemouse_teleop --ros-args -p input_source:=fake_sine
ros2 topic echo --once /hsa/teleop_target
ros2 topic hz /polyumi/target_poses_preview
```

With `-p execute:=true` (still no robot) it publishes `/polyumi/target_poses_traj` and warns every
two seconds that nothing is subscribed. That warning is the safety net you want to see.

## Run

Useful while anything runs: `ros2 topic list`, `ros2 topic echo /hsa/teleop_target`,
`ros2 topic hz /polyumi/target_poses_traj`, `ros2 node list`, `ros2 param dump /spacemouse_teleop`,
`ros2 service list`, `ros2 run tf2_ros tf2_echo fer_link0 hsa_tcp`.

### Teleop on the arm

The station runs `nuc/launch/bringup.launch.py robot:=fer` (franka_bringup with `arm_id:=fer`, the
wrist payload, `fer_arm_controller`, the static `hsa_tcp` transform) and
`nuc/launch/inference.launch.py robot:=fer` (the home service and the streaming controller,
`execute_arm:=true` to let it drive). `./hsa_session.sh` syncs `nuc/` to the station, builds the
controller there and opens every pane; `docs/hsa-arm-inference.md` is the full procedure.

### Which robot

Every launch file here takes `robot:=fer|fr3` (default `$HSA_ROBOT`, else `fer`), which loads
`config/robots/<robot>.yaml`: the base and TCP frames, the only thing the nodes need to know about
the arm. With `ros2 run`, add that file after the node's own:

    ros2 run hsa_ros2 demo_recorder --ros-args --params-file config/recorder.yaml --params-file config/robots/fr3.yaml

`docs/hsa-arm-inference.md` ("Running on the CRB FR3 instead") covers the arm side.

1. Bringup on the station, then the inference launch with `execute_arm:=true`;
   `ros2 control list_controllers` there must show `polyumi_cartesian_impedance_controller active`.
2. On this laptop, `source setup_franka_env.sh`; `ros2 run tf2_ros tf2_echo fer_link0 hsa_tcp` prints a
   pose, so TF crosses the cable.
3. Preview first, in its own terminal (the keyboard needs a TTY):
   `ros2 run hsa_ros2 spacemouse_teleop --ros-args --params-file <config/teleop.yaml>`
   and watch `/polyumi/target_poses_preview` in Foxglove or RViz (`ros2 launch hsa_ros2 rviz.launch.xml`).
4. Then `-p execute:=true`, low speeds first (`-p max_pos_speed:=0.1`). The target seeds from the
   measured TCP, so the arm does not jump when execution starts.

Before the first real session: measure what `nuc/hsa_calib.py` marks ESTIMATE (plate height,
payload), run `tcp_pivot_test`, and measure the latencies listed under Inference below.

`ros2 launch hsa_ros2 teleop.launch.xml execute:=true` does the same without a TTY, for headless
tests or when the recorder is driven another way. Keys: `c` start episode, `s` stop, Backspace drop,
Space stage+1, `n` next plan row, `p` print pose, `r` re-seed from TF, `x` toggle wrist stiff/soft,
`q` quit.

### Recorder

```bash
ros2 run hsa_ros2 demo_recorder --ros-args -p output_dir:=~/hsa_data/peg_v1 --params-file .../recorder.yaml
ros2 service call /hsa_recorder/start_episode std_srvs/srv/Trigger "{}"   # or press c in the teleop
ros2 service call /hsa_recorder/stop_episode  std_srvs/srv/Trigger "{}"   # or s
ros2 service call /hsa_recorder/export_zip    std_srvs/srv/Trigger "{}"   # -> <output_dir>/<name>.zarr.zip
```

**The camera frame is the clock.** Every stored step is one camera frame: the image goes through
`polyumi_ros2.camera_preproc.resize_camera0_rgb` (the same function the policy client uses, so
training and deployment pixels match); the TCP pose is looked up in TF at the frame's *capture*
instant (`stamp - latency.camera`), as the client's `_lookup_agent_pos` does; the newest teleop
target issued before that instant is the `action`; the stage counter, the wrist servo state
(interpolated) and the Franka external wrench (held) come along when enabled. On `stop_episode` the
episode is appended to `<output_dir>/replay_buffer.zarr` plus a JSON sidecar under `episodes/`
(plan row, condition, wrist set and hours, TF drops, key list); restarting the node appends to the
same dataset. `export_zip` (the service, or `export_dataset --zip`) packs it into the `.zarr.zip`
that `train_policy.sh` takes. The key layout is the module docstring of `dataset_writer.py`; the
design decisions (why `action` is recorded, why the extra keys are harmless) are in
`~/hsa-ws/HSA_UMI_COLLECTION_PLAN.md`.

Per-episode labels, both written to the sidecar and never interpreted by the recorder:

- **The plan** (`plan:` a CSV with any columns, one row per episode). Rows advance when an episode is
  saved (a dropped episode keeps its row), `n` skips a row, and the log names the next row after
  every save. Example: `wrist,stiffness,cell` / `rigid,n/a,A1` / `articulated,soft,C4`.
- **The condition**, a free-form tag set live and applied to every episode from then on:
  `ros2 param set /hsa_recorder condition 'dagger_round=2'`.

`ros2 run hsa_ros2 export_dataset <dir> --episodes` prints the ledger: one line per episode with its
steps, duration, condition and every plan column.

### Collection view

`collect_ui` shows the camera with the demo number, the recorder's state, the external force and
the frame's capture timestamp drawn on it, and takes the recorder's keys from the window:

```bash
ros2 run hsa_ros2 collect_ui --ros-args -p latency.camera:=0.03    # the same value as recorder.yaml
```

`c` starts an episode, `s` stops and saves it, backspace twice drops it, `q` closes the view. The
force shown is the wrench sample the recorder pairs with that frame (the newest stamp at or before
the capture instant) with its age next to it. The recorder publishes its state as JSON on
`/hsa_recorder/status` and writes each episode's pairing ages to the sidecar.

### Camera

```bash
ros2 launch hsa_ros2 camera.launch.xml
ros2 topic hz /camera/color/image_raw        # expect ~30 Hz, encoding rgb8
```

### Inference on the laptop

```bash
ros2 launch hsa_ros2 inference.launch.xml                          # log + preview only
ros2 launch hsa_ros2 inference.launch.xml execute_motion:=true     # publishes chunks to the arm
ros2 launch hsa_ros2 inference.launch.xml inference_server_url:=http://lamb.mech.northwestern.edu:8002/predict_cartesian/
ros2 run hsa_ros2 diag_stats --ros-args -p seconds:=40
```

The policy server is wherever `inference_server_url` points: this laptop's `polyumi-dp` container
by default, or lamb. `latency.gopro` (and `recorder.yaml`'s equal `latency.camera`, 0.030 s) and
`latency.arm_exec` (0.04 s) in `config/inference.yaml` are ESTIMATES not yet measured on the FER:
`latency_probe mode:=camera` / `mode:=arm` on the first session.

To serve a checkpoint on this laptop (Docker with the NVIDIA container toolkit; the host stays
conda-free) and run the whole loop with no robot:

```bash
cd ~/hsa-ws/HSA-UMI && ./build_policy_image.sh                     # once, ~25 min cold
scripts/desk_test.sh /tmp/hsa_desk                                    # a recorder dataset for the normalizer
# make the checkpoint (command in make_dummy_checkpoint.py's docstring), then:
scripts/loopback_inference_test.sh ~/hsa-ws/HSA-UMI/data/dp_outputs/dummy_random.ckpt 40
```

A real checkpoint from lamb drops into the same command; only `CKPT` changes. `serve_policy.sh`
assumes rootless Docker; on rootful Docker pass `USER_FLAG="--user $(id -u):$(id -g)"` (the loopback
script does), and inside the container use plain `python`, not `micromamba run`, when running as
your own uid. Measured 2026-09-29 on the dev laptop (RTX 500 Ada 4 GB, random-weights checkpoint,
50 inferences): forward pass 209 ms mean / 217 ms p95, request overhead 12 / 25 ms, whole request
222 / 238 ms, `obs_age_s` 385 / 417 ms (with `latency.gopro` still at 109 ms), and 11 of 16
waypoints per chunk still in the future when it arrived.

## Figures

`hsa_ros2/figures.py` draws every figure from plain arrays; the scripts only load the data.

```bash
scripts/plot_dataset.py ~/hsa_data/session --box config/teleop.yaml --3d   # workspace coverage of a dataset
scripts/record_rollout.sh peg_01                                          # bag a rollout (policy or teleop), Ctrl-C to stop
scripts/plot_rollout.py ~/hsa_data/rollouts/peg_01                        # chunks vs executed, latency timeline, tracking and force
scripts/plot_outcomes.py ledger.csv                                       # start offsets by outcome, insertion depth, Wilson interval
scripts/check_sync.py ~/hsa_data/session --episode 0                      # tap test: the frames around the force spike
```

The ledger's columns are those of `config/eval_ledger_example.csv`. `plot_rollout.py` needs the
ROS environment (rosbag2 and TF); the others need only numpy, matplotlib and zarr.

### Keeping force and frames in sync

Both streams carry header stamps: the camera's in the laptop's clock, the wrench's in the
station's, and chrony keeps the station on the laptop's clock (`ssh station chronyc tracking`
should show an offset well under a millisecond). The recorder and `collect_ui` pair them by those
stamps, never by arrival: the newest wrench sample at or before the frame's capture instant
(`stamp - latency.camera`). That sample's age is on screen while collecting and in each episode's
sidecar as `wrench_sample_age_s`. `check_sync.py` is the end-to-end check: record a short episode
with the wrench enabled, tap the tool where the camera sees it, and the tap must be visible in the
outlined frame or its neighbour (one frame is 33 ms). If it is consistently one frame late, raise
`latency.camera`.
