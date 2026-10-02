#!/usr/bin/env bash
# The deployment topology with the arm OFF: the station runs the FER bringup on mock hardware and
# the inference launch (home service) from its synced nuc/; this laptop runs the RealSense, the
# policy container on loopback and policy_client_node, with TF crossing the cable. Prints the same
# diagnostics as loopback_inference_test.sh so the two compare, then homes the mock arm over the cable.
#
#     scripts/station_mock_test.sh            # 40 s of diagnostics
#     SECS=90 scripts/station_mock_test.sh
#
# Needs: key ssh to `station` (ssh-copy-id msr@192.168.18.1 once), ./deploy_station.sh done, the
# polyumi-dp image built here, and a checkpoint at data/dp_outputs/dummy_random.ckpt (or CKPT=...).
# Logs: ~/.local/state/hsa/station_mock_test/ here, ~/hsa-mock-*.log on the station.
set +u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
LOG="${XDG_STATE_HOME:-$HOME/.local/state}/hsa/station_mock_test"; mkdir -p "$LOG"; rm -f "$LOG"/*.log
STATION=${STATION:-station}
SECS=${SECS:-40}
CKPT=${CKPT:-$REPO/data/dp_outputs/dummy_random.ckpt}
# HSA_DDS=fastdds runs both sides on FastDDS (for a station without rmw_cyclonedds_cpp).
HSA_DDS=${HSA_DDS:-fastdds}
STATION_ENV="export HSA_DDS=$HSA_DDS; source ~/HSA-UMI/nuc/station_env.sh"

cleanup() {
  kill $CLIENT_PID 2>/dev/null; sleep 3
  pkill -f '[r]ealsense2_camera_no'"de" 2>/dev/null; pkill -f '[p]olicy_client_no'"de" 2>/dev/null; pkill -f '[r]os2 laun'"ch hsa_ros2" 2>/dev/null
  docker ps -q --filter ancestor=polyumi-dp | xargs -r docker kill >/dev/null 2>&1; kill $SERVER_PID 2>/dev/null
  # The launch children too: a robot_state_publisher that outlives its launch keeps latching a
  # mock-hardware robot description, and the next REAL bringup's controller manager loads it.
  ssh -o ConnectTimeout=5 "$STATION" 'for s in hsa-mock-bringup hsa-mock-inference; do tmux kill-session -t $s 2>/dev/null; done; sleep 2; for p in "[r]os2 laun" "[r]os2_control_no" "[c]ontroller_manager/spaw" "[f]er_home_servi" "[r]obot_state_pub" "[j]oint_state_pub" "[s]tatic_transform_pub"; do pkill -f "$p"; done; sleep 1; echo "station leftovers: $(pgrep -af "[r]os2|[r]obot_state_pub|[j]oint_state_pub|[s]tatic_transform_pub" | wc -l)"; true' 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

# Start a command on the station in the background: a remote tmux session when tmux is there,
# nohup otherwise. $1 = name, $2 = the command (run through a login shell so ~/.bashrc applies).
remote_bg() {
  if ssh -o ConnectTimeout=5 "$STATION" 'command -v tmux >/dev/null'; then
    ssh -o ConnectTimeout=5 "$STATION" "tmux new-session -d -s $1 \"bash -lc '$2 2>&1 | tee ~/$1.log'\""
  else
    ssh -o ConnectTimeout=5 "$STATION" "nohup bash -lc '$2' > ~/$1.log 2>&1 &"
  fi
}

echo "### 0. station facts"
ssh -o ConnectTimeout=5 "$STATION" "hostname; uname -r; echo ROS: \$(ls /opt/ros | tr '\n' ' '); echo rmw: \$(ls /opt/ros/*/lib/ 2>/dev/null | grep -oE 'librmw_[a-z_]+_cpp\.so' | sed 's/lib//;s/\.so//' | sort -u | tr '\n' ' '); command -v tmux >/dev/null && echo 'tmux: ok' || echo 'tmux: missing (nohup fallback)'; test -f ~/franka_ws/install/setup.bash && echo 'controller overlay: built' || echo 'controller overlay: MISSING -> ./deploy_station.sh'; echo 'chrony:'; chronyc sources 2>/dev/null | tail -2" 2>&1 || { echo "cannot ssh to $STATION -- ssh-copy-id msr@192.168.18.1 first"; exit 1; }

echo; echo "### 1. policy server on this laptop (loads in the background)"
( cd "$REPO" && USER_FLAG="--user $(id -u):$(id -g)" CKPT="$CKPT" PORT=8002 ./serve_policy.sh > "$LOG/server.log" 2>&1 ) &
SERVER_PID=$!

echo "### 2. station: mock bringup, then the inference launch"
remote_bg hsa-mock-bringup "$STATION_ENV && cd ~/HSA-UMI && ros2 launch nuc/launch/fer_bringup.launch.py use_fake_hardware:=true robot_ip:=127.0.0.1"
sleep 16
remote_bg hsa-mock-inference "$STATION_ENV && cd ~/HSA-UMI && ros2 launch nuc/launch/fer_inference.launch.py home_pose_file:=/tmp/hsa_mock_home_pose.yaml"
sleep 12
echo "--- station: controllers + key log lines"
ssh -o ConnectTimeout=5 "$STATION" "bash -lc '$STATION_ENV >/dev/null 2>&1; ros2 control list_controllers 2>&1 | head -5; grep -E \"Loaded hardware|TCP |ERROR|rror:\" ~/hsa-mock-bringup.log | cut -c1-160 | head -6; grep -E \"fer_home_service|ERROR\" ~/hsa-mock-inference.log | cut -c1-160 | head -4'" 2>&1

echo; echo "### 3. laptop: DDS over the cable"
cd "$REPO"; unset VIRTUAL_ENV
source setup_franka_env.sh > "$LOG/env.log" 2>&1; grep -E "env.handell|RMW|CYCLONEDDS" "$LOG/env.log"
if [ "$HSA_DDS" = fastdds ]; then export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; unset CYCLONEDDS_URI; echo "HSA_DDS=fastdds: RMW=$RMW_IMPLEMENTATION"; fi
[ -f ~/ros2_overlay_ws/install/setup.bash ] && source ~/ros2_overlay_ws/install/setup.bash
echo "--- station nodes seen from here:"; timeout 10 ros2 node list 2>/dev/null | grep -E "controller_manager|robot_state_publisher|fer_home_service" | head -4
# Discovery across the cable takes several seconds on a fresh participant, so the probe waits longer
# than the frames need once found; a "does not exist" here that the client later resolves is discovery, not TF.
echo "--- TF fer_link0 -> hsa_tcp over the cable (up to 25 s for discovery):"; timeout 25 ros2 run tf2_ros tf2_echo fer_link0 hsa_tcp 2>&1 | grep -m1 -E "Translation|Failure"
echo "--- services:"; timeout 10 ros2 service list 2>/dev/null | grep polyumi
echo "--- station clock:"; ssh "$STATION" 'chronyc tracking 2>/dev/null | grep -E "System time|Reference"' 2>&1

echo; echo "### 4. waiting for the policy server"
for i in $(seq 150); do curl -sf localhost:8002/health >/dev/null 2>&1 && break; sleep 2; done; curl -s localhost:8002/health | cut -c1-80; echo

echo; echo "### 5. laptop: RealSense + policy_client_node (execute_motion:=false), $SECS s of diagnostics"
ros2 launch hsa_ros2 inference.launch.xml inference_server_url:=http://localhost:8002/predict_cartesian/ execute_motion:=false > "$LOG/client.log" 2>&1 &
CLIENT_PID=$!
sleep 15
echo -n "--- camera: "; timeout 6 ros2 topic hz /camera/color/image_raw 2>&1 | grep -m1 "average rate"
ros2 run hsa_ros2 diag_stats --ros-args -p seconds:=$SECS 2>&1 | grep -v '^\['
echo "--- client warnings (deduplicated):"; grep -E "WARN|ERROR" "$LOG/client.log" | grep policy_client | sed 's/.*\] //' | sort | uniq -c | sort -rn | head -5

echo; echo "### 6. home the mock arm on the station from this laptop"
timeout 40 ros2 service call /polyumi/home std_srvs/srv/Trigger "{}" 2>&1 | grep -E "success|message"
timeout 20 ros2 service call /polyumi/set_home std_srvs/srv/Trigger "{}" 2>&1 | grep -E "success|message"
echo "--- station home-service log:"; ssh "$STATION" 'grep -E "MOVING|Homed|set_home|declined" ~/hsa-mock-inference.log | cut -c1-140 | tail -4' 2>&1
echo "### done; laptop logs in $LOG, station logs in ~/hsa-mock-*.log on $STATION"
