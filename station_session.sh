#!/usr/bin/env bash
# Station side of rigid demo collection as one tmux window: bringup, inference stack, controller check.
#
#     ./station_session.sh                  # deploy, then create (or re-attach to) tmux session 'station'
#     SKIP_DEPLOY=1 ./station_session.sh    # ...without re-syncing and rebuilding on the station first
#     HSA_TOOL=rigid ./station_session.sh   # another payload (default wrist = the HSA wrist with the key, 0.933 kg)
#     HOME_POSE=key_in_hole_rigid ./station_session.sh   # another taught home pose (default key_in_hole_<tool>)
#     ./station_session.sh --kill           # ^C every pane, then close the session
#
# Every pane is `ssh -t station` with station_env.sh sourced. Bringup is RUN (moves nothing);
# the inference stack and the controller check are only PRE-TYPED: press Enter once bringup is up.
# The station has no tmux, so the panes are plain ssh: a laptop sleep kills bringup with them.
# Bringup output is tee'd to ~/.local/state/hsa/<robot>_bringup_<date>.log on the station.

set -euo pipefail

SESSION="${SESSION:-station}"
ROBOT="${HSA_ROBOT:-fer}"
TOOL="${HSA_TOOL:-wrist}"
# One taught pose per tool: the wrist's TCP is longer than the rigid mount's, so the rigid pose does not fit.
# Teach it once: hand-guide, then `ros2 service call /polyumi/set_home std_srvs/srv/Trigger "{}"` (record_session's home pane).
HOME_POSE="${HOME_POSE:-key_in_hole_$TOOL}"
STATION_SSH_HOST="${STATION_SSH_HOST:-station}"
STATION_REPO="${STATION_REPO:-~/HSA-UMI}"
SHELL_SETTLE_S="${SHELL_SETTLE_S:-5}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v tmux >/dev/null || { echo "error: tmux is not installed (sudo apt install tmux)" >&2; exit 1; }

if [ "${1:-}" = --kill ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    # ^C first so ros2 launch runs its shutdown and releases the FCI, then close.
    tmux list-panes -s -t "$SESSION" -F '#{pane_id}' | xargs -r -I{} tmux send-keys -t {} C-c
    echo "Waiting 6s for shutdown ..."; sleep 6
    tmux kill-session -t "$SESSION" && echo "Killed session '$SESSION'."
  fi
  exit 0
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--kill]" >&2; exit 2
fi

# From inside tmux (e.g. a pane of the other session) attach would nest; switch the client instead.
attach() { if [ -n "${TMUX:-}" ]; then exec tmux switch-client -t "$SESSION"; else exec tmux attach -t "$SESSION"; fi; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' exists; attaching."
  attach
fi

if [ "${SKIP_DEPLOY:-0}" = 1 ]; then
  echo "SKIP_DEPLOY=1 -- leaving the station tree as it is."
else
  echo "==> Deploying to $STATION_SSH_HOST via ./deploy_station.sh (robot $ROBOT) ..."
  if ! (cd "$REPO_DIR" && HSA_ROBOT="$ROBOT" ./deploy_station.sh "$STATION_SSH_HOST" "$STATION_REPO"); then
    echo "WARNING: deploy_station.sh failed -- the station may run stale nuc/ code or a stale controller." >&2
  fi
fi

LOG_DIR='"${XDG_STATE_HOME:-$HOME/.local/state}"/hsa'
STATION_ENV="export HSA_ROBOT=$ROBOT HSA_TOOL=$TOOL; source $STATION_REPO/nuc/station_env.sh && cd $STATION_REPO"
BRINGUP="mkdir -p $LOG_DIR && ros2 launch nuc/launch/bringup.launch.py robot:=$ROBOT 2>&1 | { trap '' INT; tee -a $LOG_DIR/${ROBOT}_bringup_\$(date +%F).log; }"
INFERENCE="ros2 launch nuc/launch/inference.launch.py robot:=$ROBOT home_pose:=$HOME_POSE execute_arm:=true"
CHECK="ros2 control list_controllers"

# Bringup on the left, inference top right, check bottom right. Panes are addressed by ID.
# Mouse: click a pane to focus it, wheel to scroll, drag a border to resize. history-limit only applies to
# panes created after it is set, so the session starts with a throwaway window that is closed once the real ones exist.
tmux new-session -d -s "$SESSION" -n _init -c "$REPO_DIR"
tmux set-option -t "$SESSION" mouse on
tmux set-option -g history-limit "${HISTORY_LIMIT:-50000}"
P_BRINGUP="$(tmux new-window -t "$SESSION:" -n station -P -F '#{pane_id}' -c "$REPO_DIR")"
tmux kill-window -t "$SESSION:_init"
P_INFERENCE="$(tmux split-window -t "$P_BRINGUP" -h -P -F '#{pane_id}' -c "$REPO_DIR")"
P_CHECK="$(tmux split-window -t "$P_INFERENCE" -v -P -F '#{pane_id}' -c "$REPO_DIR")"
tmux select-pane -t "$P_BRINGUP" -T bringup
tmux select-pane -t "$P_INFERENCE" -T inference
tmux select-pane -t "$P_CHECK" -T check
tmux set-option -w -t "$P_BRINGUP" pane-border-status top

for p in "$P_BRINGUP" "$P_INFERENCE" "$P_CHECK"; do
  tmux send-keys -t "$p" "ssh -t $STATION_SSH_HOST" C-m
done
sleep "$SHELL_SETTLE_S"
for p in "$P_BRINGUP" "$P_INFERENCE" "$P_CHECK"; do
  tmux send-keys -t "$p" "$STATION_ENV" C-m
done
# Keys sent while station_env.sh is still printing get echoed into its output as well.
sleep "${ENV_SETTLE_S:-3}"
tmux send-keys -t "$P_BRINGUP" "$BRINGUP" C-m
# No C-m: read the line, press Enter.
tmux send-keys -t "$P_INFERENCE" "$INFERENCE"
tmux send-keys -t "$P_CHECK" "$CHECK"
tmux select-pane -t "$P_BRINGUP"

cat <<EOF

Session '$SESSION' is up: robot:=$ROBOT, HSA_TOOL=$TOOL, home_pose:=$HOME_POSE.
  1. bringup (left)      already running; check it prints the payload mass (rigid: 0.179 kg)
  2. inference (top)     press Enter once bringup is up (execute_arm:=true)
  3. check (bottom)      press Enter: the impedance controller must be active

After a safety stop: Desk clear, ^C + Up + Enter in bringup, then in inference.
tmux: C-b o next pane, C-b arrows move, C-b z zoom a pane, C-b d detach. Stop: ./station_session.sh --kill

EOF
attach
