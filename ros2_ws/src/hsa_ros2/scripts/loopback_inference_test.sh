#!/usr/bin/env bash
# End-to-end inference loop on ONE machine, no robot: the polyumi-dp container serving a checkpoint
# on loopback, a static transform standing in for the arm, fake_camera for the RealSense, and the
# real polyumi_ros2 policy_client_node in log-only mode. Prints the client's round-trip statistics,
# i.e. what Option 2 (serve on the laptop) costs per inference.
#
#   scripts/loopback_inference_test.sh <checkpoint.ckpt> [seconds]
#   SERVER_URL=http://lamb.mech.northwestern.edu:8002 scripts/loopback_inference_test.sh - [seconds]
#       No container is started here; the client is pointed at an already-running server (Option 1)
#       so the same diag_stats numbers compare the two.
#
# Needs: the polyumi-dp image (./build_policy_image.sh), the ros2_ws built, and polyumi_inference
# installed for ROS's python (pip install --user --break-system-packages --no-deps -e inference_server/).
# 8002 is serve_policy.sh's default port.
set +u
CKPT=${1:-}; SECONDS_TO_RUN=${2:-40}
[ -z "${SERVER_URL:-}" ] && [ -z "$CKPT" ] && { echo "usage: $0 <checkpoint.ckpt> [seconds]   or   SERVER_URL=http://host:8002 $0 - [seconds]"; exit 2; }
WS=$(cd "$(dirname "$0")/../../.." && pwd); REPO=$(cd "$WS/.." && pwd)
unset VIRTUAL_ENV; source "/opt/ros/${ROS_DISTRO:-kilted}/setup.bash"
# foxglove_msgs: apt (ros-<distro>-foxglove-msgs) or the source overlay built from the Foxglove SDK.
[ -f "$HOME/ros2_overlay_ws/install/setup.bash" ] && source "$HOME/ros2_overlay_ws/install/setup.bash"
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-79} ROS_LOCALHOST_ONLY=1
LOG=${LOG_DIR:-/tmp/hsa_loopback}; mkdir -p "$LOG"

SERVER_PID=""
if [ -n "${SERVER_URL:-}" ]; then
    echo "### using the server already running at $SERVER_URL"
    curl -sf "$SERVER_URL/health" || { echo "no answer from $SERVER_URL/health (firewall? try: ssh -L 8002:localhost:8002 <host> and SERVER_URL=http://localhost:8002)"; exit 1; }
else
    SERVER_URL=http://localhost:8002
    echo "### serving $CKPT on :8002 (rootful docker -> run as $(id -u):$(id -g))"
    ( cd "$REPO" && USER_FLAG="--user $(id -u):$(id -g)" CKPT="$CKPT" PORT=8002 ./serve_policy.sh > "$LOG/server.log" 2>&1 ) &
    SERVER_PID=$!
    for i in $(seq 120); do curl -sf "$SERVER_URL/health" > /dev/null 2>&1 && break; sleep 2; done
    curl -s "$SERVER_URL/health" || { echo "server never became healthy; see $LOG/server.log"; kill $SERVER_PID; exit 1; }
fi
echo

ros2 run tf2_ros static_transform_publisher --x 0.5 --y 0 --z 0.3 --roll 3.14159 --pitch 0 --yaw 0 \
    --frame-id fer_link0 --child-frame-id hsa_tcp > "$LOG/tf.log" 2>&1 &
TF_PID=$!
ros2 run hsa_ros2 fake_camera > "$LOG/camera.log" 2>&1 &
CAM_PID=$!
sleep 2
ros2 run polyumi_ros2 policy_client_node --ros-args \
    --params-file "$WS/install/polyumi_ros2/share/polyumi_ros2/config/inference.yaml" \
    -p inference_server_url:="$SERVER_URL/predict_cartesian/" \
    -p image_topic:=/camera/color/image_raw -p base_frame:=fer_link0 -p eef_frame:=hsa_tcp \
    -p execute_motion:=false -p require_gripper_state:=false -p max_image_age_s:=0.3 \
    > "$LOG/client.log" 2>&1 &
CLIENT_PID=$!
sleep 8
echo "### client log so far:"; grep -v -E 'Waiting|filling' "$LOG/client.log" | tail -6
echo "### collecting diagnostics for $SECONDS_TO_RUN s"
ros2 run hsa_ros2 diag_stats --ros-args -p seconds:=$SECONDS_TO_RUN 2>&1 | grep -v '^\['

kill $CLIENT_PID $CAM_PID $TF_PID 2>/dev/null
if [ -n "$SERVER_PID" ]; then
    docker ps -q --filter ancestor=polyumi-dp | xargs -r docker kill > /dev/null 2>&1
    kill $SERVER_PID 2>/dev/null
fi
wait 2>/dev/null
echo "### client action lines (last 3):"; grep 'action chunk' "$LOG/client.log" | tail -3
echo "### logs in $LOG"
