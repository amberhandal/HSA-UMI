#!/usr/bin/env bash
# Record a rollout (policy or teleop) for scripts/plot_rollout.py: the commanded chunks, TF,
# the client's diagnostics, the teleop target, the recorder status (episode boundaries), the
# external wrench, and the wrist stack when it runs (wrist state, ODrive telemetry, joint commands,
# plate pose, tag detections, the baseplate camera's /compressed stream only: the raw 1280x800
# frames at 120 Hz are 360 MB/s).
#
#     scripts/record_rollout.sh [name]      # -> ~/hsa_data/rollouts/<name or timestamp>, Ctrl-C to stop
#     scripts/record_rollout.sh data/key_in_hole/rigid/v1/eval/ep050/r01   # a path (has a /) is used as given
set -euo pipefail
name=${1:-$(date +%Y-%m-%d_%H-%M-%S)}
case "$name" in
  */*) out=$name ;;
  *) out=~/hsa_data/rollouts/$name ;;
esac
if [ -e "$out" ]; then echo "error: $out exists; pick a new name" >&2; exit 1; fi
mkdir -p "$(dirname "$out")"
exec ros2 bag record -o "$out" -e '^/polyumi/|^/tf$|^/tf_static$|^/hsa/|^/hsa_recorder/status$|^/franka_robot_state_broadcaster/external_wrench|^/hsa_odrive_control/|^/odrive_axis[0-9]+/(controller_status|odrive_status)$|^/platform_|^/plate_mount|^/apriltag/detections$|^/tracking_cam/(camera_info|image_raw/compressed)$'
