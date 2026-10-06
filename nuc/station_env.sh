# shellcheck shell=bash
# Source this (do NOT execute) in every shell on the box that drives the arm:
#
#     source ~/HSA-UMI/nuc/station_env.sh                    # the station's FER (default)
#     export HSA_ROBOT=fr3; source ~/HSA-UMI/nuc/station_env.sh   # the CRB NUC's FR3
#
# Sources the Franka workspace, the overlay with the streaming impedance controller (built by
# deploy_station.sh) and the DDS settings that let the client's nodes find the arm, and exports
# HSA_ROBOT so `ros2 launch nuc/launch/{bringup,inference}.launch.py` default to this box's arm.
# A file in the synced tree rather than ~/.bashrc, so `ssh host 'source ... && cmd'` gets the same
# environment an interactive shell does (a bare `ssh host 'cmd'` skips ~/.bashrc). Idempotent.

_hsa_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${HSA_ROBOT:=fer}"
export HSA_ROBOT
case "$HSA_ROBOT" in
  # The station carries two NU-MSR workspaces. /opt/ros/franka (Jazzy) is stale: its hardware plugin
  # predates the ros2_control 4.38 upgrade and dies in on_init with "Internal state_handle is NULL".
  # /opt/ros/franka_kilted matches its ros2_control and the laptop's distro.
  fer)
    : "${STATION_FRANKA_PREFIX:=/opt/ros/franka_kilted}"
    : "${STATION_FRANKA_WS:=$HOME/franka_ws}"
    # fastdds: this station has no rmw_cyclonedds_cpp, and FastDDS needs no config file on either side.
    : "${HSA_DDS:=fastdds}"
    : "${HSA_CYCLONEDDS_XML:=${_hsa_here}/config/cyclonedds_station.xml}"
    ;;
  # The CRB NUC (Humble): franka_ros2, their streaming controller and the gripper drivers are in their
  # ~/franka_ws. Ours builds into a separate overlay sourced on top, so their workspace is never touched.
  # Their CycloneDDS config already lists the direct link to lamb.
  fr3)
    : "${STATION_FRANKA_PREFIX:=$HOME/franka_ws}"
    : "${STATION_FRANKA_WS:=$HOME/hsa_franka_ws}"
    : "${HSA_DDS:=cyclonedds}"
    : "${HSA_CYCLONEDDS_XML:=$HOME/franka_ws/config/cyclonedds.xml}"
    ;;
  *)
    echo "[station_env] ERROR: HSA_ROBOT must be fer or fr3, not '$HSA_ROBOT'" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# --- The Franka workspace ---
_hsa_distro="$(grep -o '/opt/ros/[a-z]*' "${STATION_FRANKA_PREFIX}/install/setup.sh" 2>/dev/null | grep -v "$STATION_FRANKA_PREFIX" | head -1 | sed 's#/opt/ros/##')"
if [ -n "${ROS_DISTRO:-}" ] && [ -n "$_hsa_distro" ] && [ "$ROS_DISTRO" != "$_hsa_distro" ]; then
  echo "[station_env] ERROR: this shell already has ROS $ROS_DISTRO sourced, but ${STATION_FRANKA_PREFIX} is $_hsa_distro." >&2
  echo "              Open a fresh terminal (or run: exec bash) and source this file again." >&2
  return 1 2>/dev/null || exit 1
fi
case ":${AMENT_PREFIX_PATH:-}:" in
  *":${STATION_FRANKA_PREFIX}/install"*) ;;
  *)
    if [ -f "${STATION_FRANKA_PREFIX}/install/setup.bash" ]; then
      # shellcheck disable=SC1091
      source "${STATION_FRANKA_PREFIX}/install/setup.bash"
    else
      echo "[station_env] WARNING: ${STATION_FRANKA_PREFIX}/install/setup.bash not found -- is this the $HSA_ROBOT box?" >&2
    fi
    ;;
esac

# --- The overlay with franka_streaming_impedance_controller (built by deploy_station.sh) ---
if [ -f "${STATION_FRANKA_WS}/install/setup.bash" ]; then
  case ":${AMENT_PREFIX_PATH:-}:" in
    *":${STATION_FRANKA_WS}/install"*) ;;
    # shellcheck disable=SC1091
    *) source "${STATION_FRANKA_WS}/install/setup.bash" ;;
  esac
else
  echo "[station_env] WARNING: ${STATION_FRANKA_WS}/install not built -- run ./deploy_station.sh from the laptop" >&2
fi

# --- DDS: match the client ---
# Whatever this says, the client must say the same: setup_franka_env.sh decides it on that side
# (config/env.<host>.sh).
export ROS_DOMAIN_ID=0
if [ "$HSA_DDS" = cyclonedds ] && ! ls "/opt/ros/${ROS_DISTRO:-jazzy}/lib/librmw_cyclonedds_cpp.so" >/dev/null 2>&1; then
  echo "[station_env] rmw_cyclonedds_cpp is not installed here; using fastdds (set HSA_DDS=fastdds on the laptop too)." >&2
  HSA_DDS=fastdds
fi
case "$HSA_DDS" in
  cyclonedds)
    [ -f "$HSA_CYCLONEDDS_XML" ] || echo "[station_env] WARNING: no CycloneDDS config at $HSA_CYCLONEDDS_XML (set HSA_CYCLONEDDS_XML)" >&2
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI="file://${HSA_CYCLONEDDS_XML}"
    ;;
  fastdds)
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    unset CYCLONEDDS_URI
    ;;
  *) echo "[station_env] ERROR: HSA_DDS must be cyclonedds or fastdds, not '$HSA_DDS'" >&2 ;;
esac

: "${HSA_TOOL:=wrist}"; export HSA_TOOL
echo "[station_env] HSA_ROBOT=$HSA_ROBOT ROS_DISTRO=${ROS_DISTRO:-?} franka workspace ${STATION_FRANKA_PREFIX}; overlay ${STATION_FRANKA_WS}"
if [ "$HSA_ROBOT" = fer ]; then
  echo "[station_env] HSA_TOOL=$HSA_TOOL (bare = nothing on the flange: TCP at the flange, zero payload)"
fi
echo "[station_env] HSA_DDS=$HSA_DDS RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID CYCLONEDDS_URI=${CYCLONEDDS_URI:-}"
unset _hsa_here
