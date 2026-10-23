#!/usr/bin/env bash
# Laptop side of policy evaluation (wrist stack, policy client, per-trial bag, home, ledger) as one tmux session.
#
#     ./rollout_session.sh ep025                                   # trials for checkpoint 'ep025' of data/key_in_hole/wrist/v1
#     ./rollout_session.sh ep045 data/key_in_hole/wrist/v1         # another checkpoint / dataset folder
#     WRIST=0 ./rollout_session.sh ep045 data/key_in_hole/rigid/v1 # no wrist window, no wrist channels on the wire
#     ./rollout_session.sh --kill                                  # ^C every pane, then close the session
#
# Everything lands in <dataset>/eval/<checkpoint>/: ledger.csv (one row per trial), bags/r_NN (one bag per
# trial), figures/. Start ./station_session.sh first (the arm) and serve the checkpoint on lamb
# (RUNBOOK §8); the client's URL is $INFERENCE_URL. Every pane is sourced (setup_franka_env.sh, which overlays
# the wrist workspace). The camera and the ledger are RUN; the wrist stack, the GUI, the client, the bag and
# home are only PRE-TYPED: press Enter.
#
#   window 'wrist'    can0 up + wrist launch | joint GUI | topic check         (WRIST=1)
#   window 'trial'    policy client (execute_motion on the line) | trial bag (next bags/r_NN) | home
#   window 'ledger'   rollout_ledger (s / f / p / u / q) | wrench throttle (1 kHz -> 100 Hz, RUN) | shell
#   window 'post'     plot_rollout on the newest bag | plot_outcomes on the ledger | shell
#
# Per trial: home (Enter; MOVES THE ARM) -> trial bag (Up + Enter) -> client (Up + Enter) -> the ledger's clock
# starts on the first chunk -> s or f in the ledger pane -> ^C the client -> ^C the bag -> home. The client has
# to be stopped before a home: /polyumi/home borrows the arm from the servo and hands it back, and a client
# still streaming would carry straight on from the home pose without a fresh /reset.

set -euo pipefail

SESSION="${SESSION:-rollout}"
ROBOT="${HSA_ROBOT:-fer}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v tmux >/dev/null || { echo "error: tmux is not installed (sudo apt install tmux)" >&2; exit 1; }

if [ "${1:-}" = --kill ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    # ^C first so the client stops streaming and the bag closes its files.
    tmux list-panes -s -t "$SESSION" -F '#{pane_id}' | xargs -r -I{} tmux send-keys -t {} C-c
    echo "Waiting 4s for shutdown ..."; sleep 4
    tmux kill-session -t "$SESSION" && echo "Killed session '$SESSION'."
  fi
  exit 0
fi

WRIST="${WRIST:-1}"
# The Arducam OV9281 by its stable path: /dev/videoN moves whenever the RealSense re-enumerates first.
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC762-video-index0}"
CKPT_LABEL="${1:?usage: $0 <checkpoint label, e.g. ep025> [dataset dir]}"
if [ "$WRIST" = 1 ]; then
  DATA="${2:-data/key_in_hole/wrist/v1}"
  CONDITION="${CONDITION:-wrist_extended}"
else
  DATA="${2:-data/key_in_hole/rigid/v1}"
  CONDITION="${CONDITION:-rigid}"
fi
DATA="${DATA%/}"
EVAL="$DATA/eval/$CKPT_LABEL"
INFERENCE_URL="${INFERENCE_URL:-http://lamb.mech.northwestern.edu:8002/predict_cartesian/}"
# Defaults true because the line is only PRE-TYPED, never run, and the station has its own execute_arm flag.
EXECUTE_MOTION="${EXECUTE_MOTION:-true}"
MAX_IMAGE_AGE_S="${MAX_IMAGE_AGE_S:-0.1}"
# The client reads a 100 Hz copy of the station's 1 kHz wrench: at 1 kHz the subscription starves its camera
# callback. topic_tools throttle makes the copy; it is RUN in the ledger window.
WRENCH_SRC="${WRENCH_SRC:-/franka_robot_state_broadcaster/external_wrench_in_base_frame}"
WRENCH_TOPIC="${WRENCH_TOPIC:-/hsa/wrench_100hz}"

attach() { if [ -n "${TMUX:-}" ]; then exec tmux switch-client -t "$SESSION"; else exec tmux attach -t "$SESSION"; fi; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' exists; attaching."
  attach
fi

mkdir -p "$REPO_DIR/$EVAL/bags"
DONE=0
[ -f "$REPO_DIR/$EVAL/ledger.csv" ] && DONE=$(( $(wc -l < "$REPO_DIR/$EVAL/ledger.csv") - 1 ))

ENV="export HSA_ROBOT=$ROBOT && source setup_franka_env.sh"
(cd "$REPO_DIR" && HSA_ROBOT="$ROBOT" bash -c 'source setup_franka_env.sh' >/dev/null) || true

WRIST_FLAGS=""
[ "$WRIST" = 1 ] && WRIST_FLAGS=" send_wrist:=true send_plate:=true send_camera1:=true"
# The client starts the RealSense itself (camera:=true, the launch default), so no separate camera pane.
CLIENT="ros2 launch hsa_ros2 inference.launch.xml robot:=$ROBOT inference_server_url:=$INFERENCE_URL execute_motion:=$EXECUTE_MOTION max_image_age_s:=$MAX_IMAGE_AGE_S send_wrench:=true wrench_topic:=$WRENCH_TOPIC$WRIST_FLAGS"
# Numbered when it runs, so Up + Enter after a ^C opens the next r_NN.
BAG_CMD="n=1; while [ -e $EVAL/bags/r_\$(printf %02d \$n) ]; do n=\$((n + 1)); done; ros2_ws/src/hsa_ros2/scripts/record_rollout.sh $EVAL/bags/r_\$(printf %02d \$n)"
HOME_CMD='ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"'
LEDGER="ros2 run hsa_ros2 rollout_ledger --ros-args -p ledger:=$REPO_DIR/$EVAL/ledger.csv -p checkpoint:=$CKPT_LABEL -p condition:=$CONDITION -p wrench_topic:=$WRENCH_TOPIC"
# modprobe first: the candleLight enumerates on USB without gs_usb loaded, and then there is no can0 to bring up.
WRIST_LAUNCH="sudo modprobe gs_usb && sudo ip link set can0 up type can bitrate 500000 && sudo ip link set can0 txqueuelen 1000; ros2 launch wrist_control wrist_control.launch.py show_window:=false video_device:=$WRIST_CAM"
WRIST_GUI="ros2 run wrist_control joint_commander_gui"
WRIST_CHECK="ros2 topic hz /hsa/wrist/state /platform_pose /tracking_cam/image_raw"
PLOT_LAST="ros2_ws/src/hsa_ros2/scripts/plot_rollout.py \$(ls -td $EVAL/bags/r_* | head -1)"
PLOT_OUTCOMES="ros2_ws/src/hsa_ros2/scripts/plot_outcomes.py $EVAL/ledger.csv --out $EVAL/figures"
THROTTLE="ros2 run topic_tools throttle messages $WRENCH_SRC 100 $WRENCH_TOPIC"

new_window() {  # $1 = window name; prints three pane IDs
  local a b c
  a="$(tmux new-window -t "$SESSION:" -n "$1" -P -F '#{pane_id}' -c "$REPO_DIR")"
  b="$(tmux split-window -t "$a" -h -P -F '#{pane_id}' -c "$REPO_DIR")"
  c="$(tmux split-window -t "$b" -v -P -F '#{pane_id}' -c "$REPO_DIR")"
  tmux set-option -w -t "$a" pane-border-status top
  echo "$a $b $c"
}
Q_PANE=(); Q_CMD=(); Q_MODE=()
pane() {  # $1 = pane, $2 = title, $3 = command, $4 = run|pretype
  tmux select-pane -t "$1" -T "$2"
  tmux send-keys -t "$1" "$ENV" C-m
  Q_PANE+=("$1"); Q_CMD+=("$3"); Q_MODE+=("$4")
}
type_commands() {
  sleep "${SHELL_SETTLE_S:-3}"
  for i in "${!Q_PANE[@]}"; do
    [ -n "${Q_CMD[i]}" ] || continue
    [ "$i" -gt 0 ] && [ "${Q_PANE[i]}" = "${Q_PANE[i-1]}" ] && sleep 1
    if [ "${Q_MODE[i]}" = run ]; then tmux send-keys -t "${Q_PANE[i]}" "${Q_CMD[i]}" C-m
    else tmux send-keys -t "${Q_PANE[i]}" "${Q_CMD[i]}"; fi
  done
}

tmux new-session -d -s "$SESSION" -n _init -c "$REPO_DIR"
tmux set-option -t "$SESSION" mouse on
tmux set-option -t "$SESSION" status-right ''
tmux set-option -g history-limit "${HISTORY_LIMIT:-50000}"
[ "$WRIST" = 1 ] && read -r P_WRIST P_GUI P_WCHECK < <(new_window wrist)
read -r P_CLIENT P_BAG P_HOME < <(new_window trial)
read -r P_LEDGER P_CHECK P_L_SHELL < <(new_window ledger)
read -r P_PLOT P_OUTCOMES P_SHELL < <(new_window post)
tmux kill-window -t "$SESSION:_init"
tmux move-window -r -t "$SESSION:"

if [ "$WRIST" = 1 ]; then
  pane "$P_WRIST" "wrist stack" "$WRIST_LAUNCH" pretype
  pane "$P_GUI" "joint gui" "$WRIST_GUI" pretype
  pane "$P_WCHECK" "wrist topics" "$WRIST_CHECK" pretype
fi
pane "$P_CLIENT" "policy client" "$CLIENT" pretype
pane "$P_BAG" "trial bag" "$BAG_CMD" pretype
pane "$P_HOME" home "$HOME_CMD" pretype
pane "$P_LEDGER" "ledger" "$LEDGER" run
pane "$P_CHECK" "wrench 100 Hz" "$THROTTLE" run
pane "$P_L_SHELL" shell "" pretype
pane "$P_PLOT" "plot newest bag" "$PLOT_LAST" pretype
pane "$P_OUTCOMES" "outcomes" "$PLOT_OUTCOMES" pretype
pane "$P_SHELL" shell "cat $EVAL/ledger.csv" pretype
type_commands

if [ "$WRIST" = 1 ]; then
  tmux select-window -t "$SESSION:wrist"; tmux select-pane -t "$P_WRIST"
else
  tmux select-window -t "$SESSION:trial"; tmux select-pane -t "$P_HOME"
fi

cat <<EOF

Session '$SESSION' is up: checkpoint $CKPT_LABEL, $EVAL ($DONE trials in the ledger), server $INFERENCE_URL.
$([ "$WRIST" = 1 ] && echo "  wrist:   wrist stack: Enter (sudo password) | joint gui: Enter -> Enable, Extend Fully (do NOT re-zero) | topics: Enter")
  trial:   policy client: Enter per trial (execute_motion:=$EXECUTE_MOTION on the line) | trial bag: Enter per trial | home: Enter (MOVES THE ARM)
  ledger:  rollout_ledger running: s success, f failure, p placement, u undo, q quit | wrench throttle running | shell
  post:    plot newest bag | outcomes figure + success rate | shell
Per trial: home -> bag (Up+Enter) -> client (Up+Enter) -> watch -> s / f -> ^C client -> ^C bag -> home.
A dry run first: edit the client line to execute_motion:=false and watch /polyumi/target_poses_preview.

tmux: C-b n/p window, C-b o next pane, C-b z zoom, C-b d detach. Stop: ./rollout_session.sh --kill

EOF
attach
