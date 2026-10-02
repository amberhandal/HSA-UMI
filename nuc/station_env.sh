# shellcheck shell=bash
# Source this (do NOT execute) in every station shell that runs the arm:
#
#     source ~/HSA-UMI/nuc/station_env.sh
#
# The station's ~/.bashrc already sources NU-MSR's /opt/ros/franka (franka_ros2 + libfranka +
# franka_description, chaining the ROS distro); this adds what HSA-UMI needs on top: the overlay with
# the streaming impedance controller, and the DDS settings that let the laptop's nodes find the arm.
# A file in the synced tree rather than ~/.bashrc, so `ssh station 'source ... && cmd'` gets the same
# environment an interactive shell does (a bare `ssh host 'cmd'` skips ~/.bashrc). Idempotent.

_hsa_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- NU-MSR's Franka workspace: the KILTED one ---
# The station carries two. /opt/ros/franka (Jazzy) is stale: its hardware plugin predates the
# ros2_control 4.38 upgrade and dies in on_init with "Internal state_handle is NULL".
# /opt/ros/franka_kilted matches its ros2_control and the laptop's distro.
: "${STATION_FRANKA_PREFIX:=/opt/ros/franka_kilted}"
_hsa_distro="$(grep -o '/opt/ros/[a-z]*' "${STATION_FRANKA_PREFIX}/install/setup.sh" 2>/dev/null | grep -v "$STATION_FRANKA_PREFIX" | head -1 | sed 's#/opt/ros/##')"
if [ -n "${ROS_DISTRO:-}" ] && [ -n "$_hsa_distro" ] && [ "$ROS_DISTRO" != "$_hsa_distro" ]; then
  echo "[station_env] ERROR: this shell already has ROS $ROS_DISTRO sourced, but ${STATION_FRANKA_PREFIX} is $_hsa_distro." >&2
  echo "              Open a fresh terminal (or run: exec bash) and source this file again." >&2
  return 1 2>/dev/null || exit 1
fi
if ! command -v ros2 >/dev/null 2>&1; then
  if [ -f "${STATION_FRANKA_PREFIX}/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${STATION_FRANKA_PREFIX}/install/setup.bash"
  else
    echo "[station_env] WARNING: ${STATION_FRANKA_PREFIX}/install/setup.bash not found -- is this the station?" >&2
  fi
fi

# --- The overlay with franka_streaming_impedance_controller (built by deploy_station.sh) ---
: "${STATION_FRANKA_WS:=$HOME/franka_ws}"
if [ -f "${STATION_FRANKA_WS}/install/setup.bash" ]; then
  case ":${AMENT_PREFIX_PATH:-}:" in
    *":${STATION_FRANKA_WS}/install"*) ;;
    # shellcheck disable=SC1091
    *) source "${STATION_FRANKA_WS}/install/setup.bash" ;;
  esac
else
  echo "[station_env] WARNING: ${STATION_FRANKA_WS}/install not built -- run ./deploy_station.sh from the laptop" >&2
fi

# --- DDS: match the laptop ---
# fastdds by default: this station has no rmw_cyclonedds_cpp and FastDDS needs no config file on
# either side. HSA_DDS=cyclonedds selects SPDP multicast + unicast peers on br0
# (config/cyclonedds_station.xml) once the package is installed. Whatever this says, the laptop
# must say the same: config/env.handell.sh carries the same default.
: "${HSA_DDS:=fastdds}"
export ROS_DOMAIN_ID=0
if [ "$HSA_DDS" = cyclonedds ] && ! ls "/opt/ros/${ROS_DISTRO:-jazzy}/lib/librmw_cyclonedds_cpp.so" >/dev/null 2>&1; then
  echo "[station_env] rmw_cyclonedds_cpp is not installed here; using fastdds (set HSA_DDS=fastdds on the laptop too)." >&2
  HSA_DDS=fastdds
fi
case "$HSA_DDS" in
  cyclonedds)
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI="file://${_hsa_here}/config/cyclonedds_station.xml"
    ;;
  fastdds)
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    unset CYCLONEDDS_URI
    ;;
  *) echo "[station_env] ERROR: HSA_DDS must be cyclonedds or fastdds, not '$HSA_DDS'" >&2 ;;
esac

: "${HSA_TOOL:=wrist}"; export HSA_TOOL
echo "[station_env] ROS_DISTRO=${ROS_DISTRO:-?} franka workspace ${STATION_FRANKA_PREFIX}; overlay ${STATION_FRANKA_WS}"
echo "[station_env] HSA_TOOL=$HSA_TOOL (bare = nothing on the flange: TCP at the flange, zero payload)"
echo "[station_env] HSA_DDS=$HSA_DDS RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID CYCLONEDDS_URI=${CYCLONEDDS_URI:-}"
unset _hsa_here
