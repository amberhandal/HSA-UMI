#!/usr/bin/env bash
# deploy_station.sh - Push the station's half of this working copy (nuc/ and the streaming
# impedance controller) to the Franka station and build the controller there.
#
# Usage: ./deploy_station.sh [ssh_host] [remote_repo_path]
#
# The station (NU-MSR's Franka workstation) runs the nuc/ launch files and the controller, which
# is C++ and must be built there against its Franka workspace. Kept separate from hsa_session.sh
# so a code-only push does not need the whole session.
#
# ssh: the station is password-only until `ssh-copy-id station`; the default host name needs this
# in ~/.ssh/config:
#     Host station
#         HostName 192.168.18.1
#         User msr

set -euo pipefail

HOST="${1:-${STATION_SSH_HOST:-station}}"
# Left unexpanded so the REMOTE shell resolves the tilde against its own $HOME.
REPO="${2:-${STATION_REPO:-~/HSA-UMI}}"
FRANKA_WS="${STATION_FRANKA_WS:-~/franka_ws}"
# The Kilted Franka workspace, not the stale Jazzy one -- see nuc/station_env.sh.
FRANKA_PREFIX="${STATION_FRANKA_PREFIX:-/opt/ros/franka_kilted}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTROLLER=external/franka_streaming_impedance_controller

# An uninitialised submodule is an EMPTY directory here, and --delete would erase the station's
# copy of it. Refuse rather than ship the deletion.
if [ -z "$(ls -A "${HERE}/${CONTROLLER}" 2>/dev/null)" ]; then
    echo "error: ${CONTROLLER} is empty (uninitialised submodule); --delete would wipe it on ${HOST}." >&2
    echo "       run: git submodule update --init ${CONTROLLER}" >&2
    exit 1
fi

echo "==> Copying nuc/ + the controller to ${HOST}:${REPO} (tar over ssh) ..."
# tar over ssh: the station has no rsync and no internet to install one. Both directories are
# replaced whole, so a file removed here is removed there; nothing else under $REPO is touched.
tar -C "${HERE}" --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='.pytest_cache' \
    -czf - nuc "${CONTROLLER}" \
  | ssh -o ConnectTimeout=10 "${HOST}" "mkdir -p ${REPO} && rm -rf ${REPO}/nuc ${REPO}/${CONTROLLER} && tar -xzf - -C ${REPO}"

# Sourcing is explicit because `ssh host 'cmd'` gets no ~/.bashrc. Only the controller package is
# symlinked: the client package is Python and runs on the laptop, so the station has no use for it.
echo "==> Building franka_streaming_impedance_controller on ${HOST} ..."
# No -u on the remote side: colcon's setup.bash reads variables it never sets (COLCON_TRACE),
# and set -u turns that into an abort before the build starts.
ssh -o ConnectTimeout=10 "${HOST}" "
    set -eo pipefail
    mkdir -p ${FRANKA_WS}/src
    ln -sfn ${REPO}/${CONTROLLER}/franka_streaming_impedance_controller \
        ${FRANKA_WS}/src/franka_streaming_impedance_controller
    source ${FRANKA_PREFIX}/install/setup.bash
    cd ${FRANKA_WS}
    # An overlay built against a different underlay (the Jazzy workspace, say) keeps stale CMake
    # caches; start clean whenever the recorded underlay differs from the one sourced now.
    if [ -f install/setup.sh ] && ! grep -q \"${FRANKA_PREFIX}\" install/setup.sh; then
        echo '    underlay changed: clean rebuild of the overlay'; rm -rf build install log
    fi
    colcon build --packages-select franka_streaming_impedance_controller \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    test -f ${REPO}/nuc/launch/fer_bringup.launch.py
    test -f ${REPO}/nuc/config/cyclonedds_station.xml
    echo '    nuc/ present, controller built'
"

echo "==> Done. NOTE: controller_manager keeps the old plugin .so mapped, so a rebuilt controller"
echo "    runs only after fer_bringup restarts. Next, on the station (or via ./hsa_session.sh):"
echo "      source ${REPO}/nuc/station_env.sh"
echo "      ros2 launch ${REPO}/nuc/launch/fer_bringup.launch.py"
