#!/usr/bin/env bash
# Record a rollout (policy or teleop) for scripts/plot_rollout.py: the commanded chunks, TF,
# the client's diagnostics, the teleop target and the external wrench.
#
#     scripts/record_rollout.sh [name]      # -> ~/hsa_data/rollouts/<name or timestamp>, Ctrl-C to stop
set -euo pipefail
out=~/hsa_data/rollouts/${1:-$(date +%Y-%m-%d_%H-%M-%S)}
mkdir -p "$(dirname "$out")"
exec ros2 bag record -o "$out" -e '^/polyumi/|^/tf$|^/tf_static$|^/hsa/|^/franka_robot_state_broadcaster/external_wrench'
