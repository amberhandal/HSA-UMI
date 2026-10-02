#!/usr/bin/env bash
# Whole-loop desk test with NO hardware: static TF (fake arm) + fake_camera + spacemouse_teleop
# on its synthetic input + demo_recorder, driven by service calls.  Records two episodes,
# discards one in progress, exports the .zarr.zip and prints the summary.
#
#   scripts/desk_test.sh [output_dir] [log_dir]        (defaults: /tmp/hsa_desk, /tmp/hsa_desk_logs)
#
# Assumes ros2_ws is built (`colcon build --packages-select hsa_ros2`).
set +u
unset VIRTUAL_ENV
ROS_DISTRO_DIR=${ROS_DISTRO_DIR:-/opt/ros/${ROS_DISTRO:-kilted}}
WS=$(cd "$(dirname "$0")/../../.." && pwd)
source "$ROS_DISTRO_DIR/setup.bash"
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-78} ROS_LOCALHOST_ONLY=1
OUT=${1:-/tmp/hsa_desk}; LOG=${2:-/tmp/hsa_desk_logs}
rm -rf "$OUT"; mkdir -p "$LOG"

printf 'wrist,stiffness,cell\nrigid,n/a,A1\narticulated,soft,C4\narticulated,stiff,F8\n' > "$OUT.plan.csv"
ros2 launch hsa_ros2 desk_test.launch.xml output_dir:="$OUT" plan:="$OUT.plan.csv" > "$LOG/launch.log" 2>&1 &
LPID=$!
sleep 6
call() { printf '%-16s ' "$1"; timeout 15 ros2 service call /hsa_recorder/$1 std_srvs/srv/Trigger "{}" 2>&1 | grep -o "success=.*"; }
echo "### services:"; ros2 service list | grep hsa_recorder
call start_episode; sleep 5; call stop_episode        # episode 0, ~150 steps at 30 Hz
call start_episode; sleep 2; call drop_episode        # discard an episode in progress
call start_episode; sleep 3; call stop_episode        # episode 1
call stop_episode                                     # not recording: expected failure
call next
call export_zip

# A launch started in the background from a script ignores SIGINT (bash disables it for
# background jobs), so stop it with SIGTERM and fall back to a hard kill.
kill -TERM $LPID; for _ in $(seq 10); do kill -0 $LPID 2>/dev/null || break; sleep 1; done
kill -9 $LPID 2>/dev/null; pkill -9 -f "hsa_ros2/lib" 2>/dev/null; pkill -9 -f "child-frame-id hsa_tcp" 2>/dev/null
wait $LPID 2>/dev/null

echo "### recorder decisions:"; grep 'hsa_recorder\]' "$LOG/launch.log" | grep -v -E '\| store|idle \|' | sed 's/^\[demo_recorder-[0-9]*\] //'
echo "### tracebacks: $(grep -c Traceback "$LOG/launch.log")"
echo "### dataset:"; ros2 run hsa_ros2 export_dataset "$OUT" 2>&1 | grep -v '^\['
echo "### episodes (sidecars):"; ros2 run hsa_ros2 export_dataset "$OUT" --episodes 2>&1 | grep -v '^\['
