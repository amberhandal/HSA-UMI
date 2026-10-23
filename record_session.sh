#!/usr/bin/env bash
# Laptop side of demo collection (wrist stack, camera, recorder with force + wrist, teleop, collect_ui, bag)
# as one tmux session.
#
#     ./record_session.sh                                   # into data/key_in_hole/wrist/v1, condition wrist_extended
#     ./record_session.sh data/key_in_hole/wrist/v2 wrist_extended   # another dataset folder / condition
#     WRIST=0 ./record_session.sh data/key_in_hole/rigid/v3 rigid    # no wrist window, no wrist streams
#     ./record_session.sh --kill                            # ^C every pane, then close the session
#
# Start ./station_session.sh first. Every pane is sourced (setup_franka_env.sh, which overlays the wrist
# workspace on this laptop). Camera and collect_ui are RUN; the wrist stack, the GUI, the recorder, teleop,
# session bag and home are only PRE-TYPED: press Enter.
#
#   window 'wrist'    can0 up + wrist launch (ODrives, state publisher, baseplate camera + tags) | joint GUI | topic check
#   window 'record'   camera | recorder (wrench + wrist + plate + wrist camera) | session bag (next bags/session_NN)
#   window 'operate'  teleop | collect_ui | shell
#   window 'home'     home (Up + Enter after every demo) | saved-demo count | shell
#   window 'post'     tap-test sync check, then end-of-session ledger / plots / zip
#
# Wrist order, before the first demo: Enter in the wrist pane (sudo asks for the password) -> Enter in the GUI
# pane -> GUI: Enable -> plate flat: Set Zero Here, Set Platform Zero (camera) if the mount changed -> Extend Fully
# (every HSA to its max_angle, 360 deg) -> the topic check shows /hsa/wrist/state and /platform_pose. The GUI's
# banner must be green (plate mounted the expected way round). Then the record window as before.

set -euo pipefail

SESSION="${SESSION:-record}"
ROBOT="${HSA_ROBOT:-fer}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v tmux >/dev/null || { echo "error: tmux is not installed (sudo apt install tmux)" >&2; exit 1; }

if [ "${1:-}" = --kill ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    # ^C first so the recorder and the bag close their files.
    tmux list-panes -s -t "$SESSION" -F '#{pane_id}' | xargs -r -I{} tmux send-keys -t {} C-c
    echo "Waiting 4s for shutdown ..."; sleep 4
    tmux kill-session -t "$SESSION" && echo "Killed session '$SESSION'."
  fi
  exit 0
fi

WRIST="${WRIST:-1}"
# The Arducam OV9281 by its stable path: /dev/videoN moves whenever the RealSense re-enumerates first.
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_OV9281_USB_Camera_UC762-video-index0}"
if [ "$WRIST" = 1 ]; then
  DATA="${1:-data/key_in_hole/wrist/v1}"
  CONDITION="${2:-wrist_extended}"
else
  DATA="${1:-data/key_in_hole/rigid/v2}"
  CONDITION="${2:-rigid}"
fi
DATA="${DATA%/}"
NAME="$(echo "$DATA" | sed 's|^data/||; s|/|_|g')"   # data/key_in_hole/rigid/v2 -> key_in_hole_rigid_v2

# From inside tmux (e.g. a pane of the other session) attach would nest; switch the client instead.
attach() { if [ -n "${TMUX:-}" ]; then exec tmux switch-client -t "$SESSION"; else exec tmux attach -t "$SESSION"; fi; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' exists; attaching."
  attach
fi

# One bag per session: the first bags/session_NN that does not exist yet.
n=1
while [ -e "$REPO_DIR/$DATA/bags/$(printf 'session_%02d' "$n")" ]; do n=$((n + 1)); done
BAG="$DATA/bags/$(printf 'session_%02d' "$n")"
EXISTING="$(find "$REPO_DIR/$DATA/episodes" -name "*.json" 2>/dev/null | wc -l)" || true

ENV="export HSA_ROBOT=$ROBOT && source setup_franka_env.sh"
# Bring the wired link up once here, so the panes find it active instead of racing `nmcli connection up`.
(cd "$REPO_DIR" && HSA_ROBOT="$ROBOT" bash -c 'source setup_franka_env.sh' >/dev/null) || true
TELEOP_YAML=ros2_ws/src/hsa_ros2/config/teleop.yaml

CAMERA="ros2 launch hsa_ros2 camera.launch.xml"
WRIST_STREAMS=""
[ "$WRIST" = 1 ] && WRIST_STREAMS=" -p wrist.enabled:=true -p plate.enabled:=true -p wrist_camera.enabled:=true"
RECORDER="ros2 run hsa_ros2 demo_recorder --ros-args --params-file ros2_ws/src/hsa_ros2/config/recorder.yaml -p output_dir:=$REPO_DIR/$DATA -p task_name:=key_in_hole -p condition:=$CONDITION -p wrench.enabled:=true$WRIST_STREAMS -p max_episode_s:=60.0"
# can0 is the candleLight USB adapter; the launch is the whole wrist (odrive_can bridges, hsa_odrive_control,
# wrist_state_publisher, baseplate camera + AprilTag + platform_pose + plate_guard). No OpenCV window: collect_ui
# shows the plate numbers and the GUI the mount banner.
# modprobe first: the candleLight enumerates on USB without gs_usb loaded, and then there is no can0 to bring up.
WRIST_LAUNCH="sudo modprobe gs_usb && sudo ip link set can0 up type can bitrate 500000 && sudo ip link set can0 txqueuelen 1000; ros2 launch wrist_control wrist_control.launch.py show_window:=false video_device:=$WRIST_CAM"
WRIST_GUI="ros2 run wrist_control joint_commander_gui"
WRIST_CHECK="ros2 topic hz /hsa/wrist/state /platform_pose /tracking_cam/image_raw"
# Numbered when it runs, not when the session starts: Up + Enter after a ^C then opens the next session_NN
# instead of failing on the folder the previous run left.
BAG_CMD="n=1; while [ -e $DATA/bags/session_\$(printf %02d \$n) ]; do n=\$((n + 1)); done; ros2_ws/src/hsa_ros2/scripts/record_rollout.sh $DATA/bags/session_\$(printf %02d \$n)"
TELEOP="ros2 run hsa_ros2 spacemouse_teleop --ros-args --params-file $TELEOP_YAML -p execute:=true -p max_pos_speed:=0.1"
COLLECT_UI="ros2 run hsa_ros2 collect_ui"
HOME_CMD='ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"'
COUNT="watch -n 2 'echo saved demos: \$(ls $DATA/episodes 2>/dev/null | wc -l)'"
SYNC="ros2_ws/src/hsa_ros2/scripts/check_sync.py $DATA --episode 0"
POST="ros2 run hsa_ros2 export_dataset \$D --csv \$D/ledger.csv && ros2_ws/src/hsa_ros2/scripts/plot_dataset.py \$D --box $TELEOP_YAML --3d --out \$D/figures && ros2 run hsa_ros2 export_dataset \$D --zip -o \$D/$NAME.zarr.zip"

# Each window: one pane left, two stacked right. Panes are addressed by the IDs tmux hands back.
new_window() {  # $1 = window name; prints three pane IDs
  local a b c
  a="$(tmux new-window -t "$SESSION:" -n "$1" -P -F '#{pane_id}' -c "$REPO_DIR")"
  b="$(tmux split-window -t "$a" -h -P -F '#{pane_id}' -c "$REPO_DIR")"
  c="$(tmux split-window -t "$b" -v -P -F '#{pane_id}' -c "$REPO_DIR")"
  tmux set-option -w -t "$a" pane-border-status top
  echo "$a $b $c"
}
# pane() sources each pane now and queues its command; type_commands() types them once the sourcing is
# done, since keys sent while setup_franka_env.sh is still printing get echoed into its output as well.
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
    # A second command for the same pane waits for the first to finish printing.
    [ "$i" -gt 0 ] && [ "${Q_PANE[i]}" = "${Q_PANE[i-1]}" ] && sleep 1
    if [ "${Q_MODE[i]}" = run ]; then tmux send-keys -t "${Q_PANE[i]}" "${Q_CMD[i]}" C-m
    else tmux send-keys -t "${Q_PANE[i]}" "${Q_CMD[i]}"; fi
  done
}

# Mouse: click a pane to focus it, wheel to scroll, drag a border to resize. history-limit only applies to
# panes created after it is set, so the session starts with a throwaway window that is closed once the real ones exist.
tmux new-session -d -s "$SESSION" -n _init -c "$REPO_DIR"
tmux set-option -t "$SESSION" mouse on
# No clock/hostname on the right, so a narrow terminal still shows every window name.
tmux set-option -t "$SESSION" status-right ''
tmux set-option -g history-limit "${HISTORY_LIMIT:-50000}"
[ "$WRIST" = 1 ] && read -r P_WRIST P_GUI P_WCHECK < <(new_window wrist)
read -r P_CAM P_REC P_BAG < <(new_window record)
read -r P_TELEOP P_UI P_OP_SHELL < <(new_window operate)
read -r P_HOME P_COUNT P_HOME_SHELL < <(new_window home)
read -r P_SYNC P_POST P_SHELL < <(new_window post)
tmux kill-window -t "$SESSION:_init"
tmux move-window -r -t "$SESSION:"

if [ "$WRIST" = 1 ]; then
  pane "$P_WRIST" "wrist stack" "$WRIST_LAUNCH" pretype
  pane "$P_GUI" "joint gui" "$WRIST_GUI" pretype
  pane "$P_WCHECK" "wrist topics" "$WRIST_CHECK" pretype
fi
pane "$P_CAM" camera "$CAMERA" run
pane "$P_REC" recorder "$RECORDER" pretype
pane "$P_BAG" "session bag" "$BAG_CMD" pretype
# A second teleop fights the first over the arm; the pgrep line shows any leftover.
pane "$P_TELEOP" teleop "pgrep -af spacemouse_teleop || echo no teleop running" run
Q_PANE+=("$P_TELEOP"); Q_CMD+=("$TELEOP"); Q_MODE+=(pretype)
pane "$P_UI" collect_ui "$COLLECT_UI" run
pane "$P_OP_SHELL" shell "" pretype
pane "$P_HOME" home "$HOME_CMD" pretype
pane "$P_COUNT" "saved demos" "$COUNT" run
pane "$P_HOME_SHELL" shell "" pretype
pane "$P_SYNC" "tap test" "$SYNC" pretype
pane "$P_POST" "end of session" "D=$DATA" run
Q_PANE+=("$P_POST"); Q_CMD+=("$POST"); Q_MODE+=(pretype)
pane "$P_SHELL" shell "ros2 run hsa_ros2 export_dataset $DATA --episodes" pretype
type_commands

if [ "$WRIST" = 1 ]; then
  tmux select-window -t "$SESSION:wrist"; tmux select-pane -t "$P_WRIST"
else
  tmux select-window -t "$SESSION:record"; tmux select-pane -t "$P_REC"
fi

cat <<EOF

Session '$SESSION' is up: $DATA (condition $CONDITION, $EXISTING demo notes already there), bag $BAG.
$([ "$WRIST" = 1 ] && echo "  wrist:   wrist stack: Enter (sudo password) | joint gui: Enter -> Enable, Set Zero Here, Extend Fully | topics check: Enter")
  record:  camera running | recorder: Enter | session bag: Enter (optional; ^C after the last demo)
  operate: teleop: Enter once 'pgrep' above it is empty | collect_ui running | shell
  home:    home: Enter, then Up + Enter after every demo (MOVES THE ARM) | saved-demo count | shell
  post:    tap test after demo 0 | end of session: ledger, plots, zip | shell
Per demo: home -> c -> 2 -> 3 -> s / f / Backspace.

tmux: C-b n/p window, C-b o next pane, C-b z zoom, C-b d detach. Stop: ./record_session.sh --kill

EOF
attach
