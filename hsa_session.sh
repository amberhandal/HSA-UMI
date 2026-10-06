#!/usr/bin/env bash
# Bring up the whole HSA inference stack -- arm box, policy server, ROS client -- as one tmux session.
#
#     ./hsa_session.sh                 # create (or re-attach to) the session, robot:=fer
#     ./hsa_session.sh robot:=fr3      # the CRB FR3 instead (also --robot fr3, or HSA_ROBOT=fr3)
#     SKIP_DEPLOY=1 ./hsa_session.sh   # ...without re-syncing and rebuilding on the arm box first
#     ./hsa_session.sh --kill-local    # tear down the LOCAL session only (the remote ones survive)
#     ./hsa_session.sh --kill          # ...and stop the remote sessions too (pass the same robot:=)
#
# robot:=fer  the station runs the arm only; this laptop runs the RealSense, policy_client_node,
#             Foxglove and, unless POLICY_HOST names a GPU box, the policy server, so inference stays
#             on loopback (docs/hsa-architecture-options.md).
# robot:=fr3  the CRB NUC (ssh jailfranka) runs the arm; lamb, cabled to it, runs the client and the
#             policy server, and this machine is only a terminal (CLIENT_HOST= runs the client here).
#
# Safe commands are RUN; anything needing a decision (which
# checkpoint, whether the arm may move) is only PRE-TYPED: edit the line, press Enter. Station
# panes run tmux ON the station (survives a laptop sleep; re-running re-attaches), so `ssh-copy-id
# station` first or every pane asks for a password. Launch output is tee'd to
# ~/.local/state/hsa/<name>_<date>.log on the machine that ran it; ~/.ros/log misses libfranka faults.

set -euo pipefail

ACTION=""
ROBOT="${HSA_ROBOT:-fer}"
while [ $# -gt 0 ]; do
  case "$1" in
    robot:=*) ROBOT="${1#robot:=}" ;;
    --robot=*) ROBOT="${1#--robot=}" ;;
    --robot) ROBOT="${2:?--robot needs fer or fr3}"; shift ;;
    --kill|--kill-local) ACTION="$1" ;;
    *) echo "usage: $0 [robot:=fer|fr3] [--kill|--kill-local]" >&2; exit 2 ;;
  esac
  shift
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHELL_SETTLE_S="${SHELL_SETTLE_S:-5}"

# Per-robot defaults; every one can still be overridden from the environment. Paths stay
# unexpanded so the REMOTE shell resolves the tilde against its own $HOME.
#   STATION_*  the box that drives the arm (nuc/ + the controller run there)
#   CLIENT_*   where the camera and policy_client_node run ("" = this machine)
#   POLICY_*   where the policy server runs ("" = this machine)
case "$ROBOT" in
  fer)
    : "${SESSION:=hsa}"
    : "${STATION_SSH_HOST:=station}"
    : "${CLIENT_HOST=}"
    : "${POLICY_HOST=}"
    ;;
  fr3)
    : "${SESSION:=hsa-fr3}"
    : "${STATION_SSH_HOST:=${NUC_SSH_HOST:-jailfranka}}"
    : "${CLIENT_HOST=${ROS_SSH_HOST:-lamb}}"
    : "${POLICY_HOST=$CLIENT_HOST}"
    ;;
  *) echo "error: robot must be fer or fr3, not '$ROBOT'" >&2; exit 2 ;;
esac
STATION_REPO="${STATION_REPO:-~/HSA-UMI}"
CLIENT_REPO="${CLIENT_REPO:-~/repos/HSA-UMI}"
POLICY_REPO="${POLICY_REPO:-~/repos/HSA-UMI}"
POLICY_PORT="${POLICY_PORT:-8002}"
# Same box as the client (or both here): loopback. Otherwise over the network.
if [ -z "$POLICY_HOST" ] || [ "$POLICY_HOST" = "$CLIENT_HOST" ]; then
  INFERENCE_URL="${INFERENCE_URL:-http://localhost:${POLICY_PORT}/predict_cartesian/}"
else
  INFERENCE_URL="${INFERENCE_URL:-http://${POLICY_HOST}:${POLICY_PORT}/predict_cartesian/}"
fi
# FR3 only: the gripper driver the arm box starts (faulhaber | hand). Empty = none: the HSA policy's
# gripper output is a placeholder, and the fingers should not follow it.
GRIPPER="${GRIPPER:-}"
# Defaults true because the line is only PRE-TYPED, never run, and the station has its own
# execute_arm flag (default false) in front of the arm.
EXECUTE_MOTION="${EXECUTE_MOTION:-true}"
# Three periods of the 30 Hz colour stream; the auto default assumes a 60 Hz camera.
MAX_IMAGE_AGE_S="${MAX_IMAGE_AGE_S:-0.1}"

REMOTE_SESSIONS=("$STATION_SSH_HOST hsa-bringup" "$STATION_SSH_HOST hsa-inference")
[ -n "$POLICY_HOST" ] && REMOTE_SESSIONS+=("$POLICY_HOST hsa-policy")
[ -n "$CLIENT_HOST" ] && REMOTE_SESSIONS+=("$CLIENT_HOST hsa-client")

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
if [ "$ACTION" = "--kill-local" ] || [ "$ACTION" = "--kill" ]; then
  KILL_GRACE_S="${KILL_GRACE_S:-6}"
  NEEDS_GRACE=0
  if [ "$ACTION" = "--kill" ]; then
    # ^C every pane of each remote session, so ros2 launch runs its shutdown (which is what
    # reports whether the FCI was released) and `docker run` stops its container.
    for hs in "${REMOTE_SESSIONS[@]}"; do
      read -r host sess <<<"$hs"
      if ssh -o ConnectTimeout=5 "$host" \
          "tmux has-session -t $sess 2>/dev/null && tmux list-panes -s -t $sess -F '#{pane_id}' \
           | xargs -r -I{} tmux send-keys -t {} C-c" 2>/dev/null; then
        NEEDS_GRACE=1
      fi
    done
  fi
  # The local panes hold the client and (by default) the policy container.
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux list-panes -s -t "$SESSION" -F '#{pane_id}' | xargs -r -I{} tmux send-keys -t {} C-c
    NEEDS_GRACE=1
  fi
  if [ "$NEEDS_GRACE" = 1 ]; then
    echo "Waiting ${KILL_GRACE_S}s for shutdown ..."; sleep "$KILL_GRACE_S"
  fi
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed local session '$SESSION'." || true
  if [ "$ACTION" = "--kill-local" ]; then
    echo "NOTE: the remote tmux sessions are still running by design. ./hsa_session.sh robot:=$ROBOT --kill stops them."
    exit 0
  fi
  for hs in "${REMOTE_SESSIONS[@]}"; do
    read -r host sess <<<"$hs"
    if ssh -o ConnectTimeout=5 "$host" "tmux kill-session -t $sess" 2>/dev/null; then
      echo "Killed remote session '$sess' on $host."
    else
      echo "No remote session '$sess' on $host (already gone, or host unreachable)."
    fi
  done
  exit 0
fi

# Already up? Just re-attach -- the normal path after a laptop sleep.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' exists; attaching."
  exec tmux attach -t "$SESSION"
fi

# Only one process may hold the FCI. On the shared NUC that may be the CRB team's own bringup.
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$STATION_SSH_HOST" 'tmux has-session -t fr3-bringup' 2>/dev/null; then
  echo "WARNING: $STATION_SSH_HOST has a live 'fr3-bringup' tmux session (fr3_session.sh). Stop it before" >&2
  echo "         running HSA bringup there: two bringups cannot both hold the FCI." >&2
fi

# ---------------------------------------------------------------------------
# Deploy: every box runs THIS working copy, not whatever it last had.
# ---------------------------------------------------------------------------
if [ "${SKIP_DEPLOY:-0}" = 1 ]; then
  echo "SKIP_DEPLOY=1 -- leaving the remote trees as they are."
else
  echo "==> Deploying to $STATION_SSH_HOST via ./deploy_station.sh (robot $ROBOT) ..."
  if ! (cd "$REPO_DIR" && HSA_ROBOT="$ROBOT" ./deploy_station.sh "$STATION_SSH_HOST" "$STATION_REPO"); then
    echo "WARNING: deploy_station.sh failed -- $STATION_SSH_HOST may run stale nuc/ code or a stale controller." >&2
  fi
  # The client box and the policy box, once each (deploy_lamb.sh syncs the tree and builds the ROS side).
  DEPLOYED=""
  for hr in "$CLIENT_HOST $CLIENT_REPO" "$POLICY_HOST $POLICY_REPO"; do
    read -r host repo <<<"$hr"
    { [ -z "$host" ] || [ "$host" = "$DEPLOYED" ]; } && continue
    echo "==> Deploying to $host via ./deploy_lamb.sh ..."
    if ! (cd "$REPO_DIR" && ./deploy_lamb.sh "$host" "$repo"); then
      echo "WARNING: deploy_lamb.sh failed -- $host may run stale code." >&2
    fi
    DEPLOYED="$host"
  done
fi

# ---------------------------------------------------------------------------
# The pane table
# ---------------------------------------------------------------------------
LOG_DIR='"${XDG_STATE_HOME:-$HOME/.local/state}"/hsa'
LOG_KEEP_DAYS="${LOG_KEEP_DAYS:-14}"

logged() {
  # $1 = short name for the log file, $2 = the command. `trap '' INT` in the tee subshell lets
  # tee read until stdout closes, so ros2 launch's shutdown lines reach the file.
  printf 'mkdir -p %s && { find %s -maxdepth 1 -name "%s_*.log" -mtime +%s -delete 2>/dev/null || true; } && %s 2>&1 | { trap "" INT; tee -a %s/%s_$(date +%%F).log; }' \
    "$LOG_DIR" "$LOG_DIR" "$1" "$LOG_KEEP_DAYS" "$2" "$LOG_DIR" "$1"
}

P_NAME=(); P_HOST=(); P_SESS=(); P_LAYOUT=(); P_RUN=(); P_PRE=()
add_pane() {
  # name  host ("" = this laptop)  remote_tmux_session ("" = none)  layout(window|split)  run  pretype
  P_NAME+=("$1"); P_HOST+=("$2"); P_SESS+=("$3"); P_LAYOUT+=("$4"); P_RUN+=("$5"); P_PRE+=("$6")
}

STATION_ENV="export HSA_ROBOT=$ROBOT; source $STATION_REPO/nuc/station_env.sh"
INFERENCE_ARGS="robot:=$ROBOT execute_arm:=true"
[ -n "$GRIPPER" ] && INFERENCE_ARGS+=" gripper:=$GRIPPER execute_gripper:=true"

# RUN: safe, moves nothing, everything else waits on it. Fails loudly if FCI is off in Desk.
add_pane station "$STATION_SSH_HOST" hsa-bringup window \
  "$STATION_ENV && cd $STATION_REPO && $(logged "${ROBOT}_bringup" "ros2 launch nuc/launch/bringup.launch.py robot:=$ROBOT")" ""

# PRETYPE: carries execute_arm, so it is yours to press Enter on.
add_pane station-inference "$STATION_SSH_HOST" hsa-inference split "" \
  "$STATION_ENV && cd $STATION_REPO && $(logged "${ROBOT}_inference" "ros2 launch nuc/launch/inference.launch.py $INFERENCE_ARGS")"

# PRETYPE: the checkpoint changes every training run. Local by default (rootful Docker: run the
# container as this user), or on POLICY_HOST over ssh.
if [ -n "$POLICY_HOST" ]; then
  add_pane policy-server "$POLICY_HOST" hsa-policy window \
    "cd $POLICY_REPO && ls -t data/dp_outputs/*/*/checkpoints/latest.ckpt 2>/dev/null | head -5" \
    "CKPT=\$(ls -t $POLICY_REPO/data/dp_outputs/*/*/checkpoints/latest.ckpt | head -1) PORT=$POLICY_PORT ./serve_policy.sh"
else
  add_pane policy-server "" "" window \
    "ls -t data/dp_outputs/*.ckpt data/dp_outputs/*/*/checkpoints/latest.ckpt 2>/dev/null | head -5" \
    "CKPT=\$(ls -t data/dp_outputs/*/*/checkpoints/latest.ckpt data/dp_outputs/*.ckpt 2>/dev/null | head -1) PORT=$POLICY_PORT USER_FLAG=\"--user \$(id -u):\$(id -g)\" ./serve_policy.sh"
fi

# PRETYPE: depends on every pane above being live, so this is the one you press Enter on last.
# setup_franka_env.sh is sourced on the line itself so the history entry carries its own DDS env;
# HSA_ROBOT goes first because the host's link settings (config/env.<host>.sh) depend on it.
if [ -z "$CLIENT_HOST" ]; then
  FOXGLOVE="${FOXGLOVE:-false}"
  # Sourced with -u off: colcon's setup scripts read variables they never set.
  if ! command -v ros2 >/dev/null 2>&1; then set +u; source /opt/ros/kilted/setup.bash 2>/dev/null || true; set -u; fi
  ros2 pkg prefix foxglove_bridge >/dev/null 2>&1 && FOXGLOVE=true
  CLIENT_CD=""; CLIENT_SESS=""; CLIENT_LAYOUT=split
else
  FOXGLOVE="${FOXGLOVE:-false}"
  CLIENT_CD="cd $CLIENT_REPO && "; CLIENT_SESS=hsa-client; CLIENT_LAYOUT=window
fi
add_pane ros-client "$CLIENT_HOST" "$CLIENT_SESS" "$CLIENT_LAYOUT" "" \
  "${CLIENT_CD}export HSA_ROBOT=$ROBOT && source setup_franka_env.sh && $(logged policy_client "ros2 launch hsa_ros2 inference.launch.xml robot:=$ROBOT inference_server_url:=$INFERENCE_URL execute_motion:=$EXECUTE_MOTION max_image_age_s:=$MAX_IMAGE_AGE_S foxglove:=$FOXGLOVE")"

# ---------------------------------------------------------------------------
# Probe, open, type
# ---------------------------------------------------------------------------
P_FRESH=(); P_CMD=(); P_PANE=()
for i in "${!P_NAME[@]}"; do
  P_FRESH[i]=1
  P_CMD[i]=""
  if [ -n "${P_HOST[i]}" ]; then
    P_CMD[i]="ssh -t ${P_HOST[i]}"
    if [ -n "${P_SESS[i]}" ]; then
      probe="$(ssh -o ConnectTimeout=5 "${P_HOST[i]}" \
        "command -v tmux >/dev/null && echo has-tmux; tmux has-session -t ${P_SESS[i]} 2>/dev/null && echo live" \
        2>/dev/null || true)"
      case "$probe" in
        *has-tmux*) P_CMD[i]="ssh -t ${P_HOST[i]} 'tmux new-session -A -s ${P_SESS[i]}'" ;;
        *) echo "WARNING: ${P_HOST[i]} has no remote tmux (or is unreachable); pane '${P_NAME[i]}' is a plain ssh." >&2 ;;
      esac
      # A live session must NOT be typed into: send-keys APPENDS to whatever is at its prompt.
      case "$probe" in *live*) P_FRESH[i]=0; echo "Re-attaching to '${P_SESS[i]}' on ${P_HOST[i]}; leaving that pane untouched." ;; esac
    fi
  fi
done

# Panes are addressed by the IDs tmux hands back (%0, %7, ...), never by "window.index" strings,
# which `pane-base-index 1` in a ~/.tmux.conf would silently shift.
LAST_WINDOW=""; LAST_PANE=""
for i in "${!P_NAME[@]}"; do
  if [ "${P_LAYOUT[i]}" = window ] && [ -z "$LAST_WINDOW" ]; then
    read -r pane win < <(tmux new-session -d -P -F '#{pane_id} #{window_id}' -s "$SESSION" -n "${P_NAME[i]}" -c "$REPO_DIR")
  elif [ "${P_LAYOUT[i]}" = window ]; then
    read -r pane win < <(tmux new-window -a -t "$LAST_WINDOW" -n "${P_NAME[i]}" -P -F '#{pane_id} #{window_id}' -c "$REPO_DIR")
  else
    pane="$(tmux split-window -t "$LAST_PANE" -h -P -F '#{pane_id}' -c "$REPO_DIR")"
    win="$LAST_WINDOW"
  fi
  P_PANE[i]="$pane"; LAST_WINDOW="$win"; LAST_PANE="$pane"
  [ -n "${P_CMD[i]}" ] && tmux send-keys -t "$pane" "${P_CMD[i]}" C-m
done
FIRST_WINDOW="$(tmux list-windows -t "$SESSION" -F '#{window_id}' | head -1)"

sleep "$SHELL_SETTLE_S"

for i in "${!P_NAME[@]}"; do
  if [ "${P_FRESH[i]}" = 1 ]; then
    [ -n "${P_RUN[i]}" ] && tmux send-keys -t "${P_PANE[i]}" "${P_RUN[i]}" C-m || true
    # No C-m: the operator reads the line and presses Enter. Every robot-moving command comes in this way.
    [ -n "${P_PRE[i]}" ] && tmux send-keys -t "${P_PANE[i]}" "${P_PRE[i]}" || true
  fi
done

tmux select-window -t "$FIRST_WINDOW"
cat <<EOF

Session '$SESSION' is up for robot:=$ROBOT. Arm box: $STATION_SSH_HOST. Client: ${CLIENT_HOST:-this machine}.
Policy server: ${POLICY_HOST:-this machine} (client URL $INFERENCE_URL).
Order to press Enter in:
  1. station, left        already running bringup (enable FCI in Desk first if it errors)
  2. station, right       the inference stack -- check execute_arm on the line before you run it
  3. policy-server        the server -- edit CKPT to the checkpoint you want
  4. ros-client           the client, last (execute_motion:=$EXECUTE_MOTION on the line)

Teach the start pose once (does NOT move), then home before every trial (MOVES THE ARM):
  ros2 service call /polyumi/set_home std_srvs/srv/Trigger "{}"
  ros2 service call /polyumi/home     std_srvs/srv/Trigger "{}"

First time on the arm: keep execute_arm:=false and execute_motion:=false, watch
/polyumi/target_poses_preview in Foxglove (ws://localhost:8765), then enable one at a time.

tmux: C-b n/p next/prev window, C-b o next pane, C-b d detach, C-b C-b prefix for the INNER tmux.
Re-attach any time with ./hsa_session.sh robot:=$ROBOT; the remote panes pick up where they left off.

EOF
exec tmux attach -t "$SESSION"
