#!/usr/bin/env bash
# deploy_station.sh - Push the arm box's half of this working copy (nuc/ and the streaming
# impedance controller) to the box that drives the arm and build the controller there.
#
# Usage: ./deploy_station.sh [ssh_host] [remote_repo_path]
#        HSA_ROBOT=fr3 ./deploy_station.sh          # the CRB NUC instead of the station
#
# The arm box runs the nuc/ launch files and the controller, which is C++ and must be built there
# against its Franka workspace. Kept separate from hsa_session.sh so a code-only push does not
# need the whole session.
#
#   fer: the NU-MSR station; overlay ~/franka_ws on /opt/ros/franka_kilted.
#   fr3: the CRB NUC; overlay ~/hsa_franka_ws on their ~/franka_ws, and the tree in ~/HSA-UMI, so
#        their checkout, their workspace and their build of the controller are never touched.
#
# ssh: the station is password-only until `ssh-copy-id station`; the default host names need
# entries in ~/.ssh/config:
#     Host station
#         HostName 192.168.18.1
#         User msr
#     Host jailfranka          # the CRB NUC; ask the CRB team for the address and account

set -euo pipefail

ROBOT="${HSA_ROBOT:-fer}"
# Paths stay unexpanded so the REMOTE shell resolves them against its own $HOME; the defaults
# match nuc/station_env.sh.
case "$ROBOT" in
    # The Kilted Franka workspace, not the stale Jazzy one -- see nuc/station_env.sh.
    fer) DEFAULT_HOST=station; DEFAULT_WS='~/franka_ws'; DEFAULT_PREFIX=/opt/ros/franka_kilted ;;
    fr3) DEFAULT_HOST=jailfranka; DEFAULT_WS='~/hsa_franka_ws'; DEFAULT_PREFIX='$HOME/franka_ws' ;;
    *) echo "error: HSA_ROBOT must be fer or fr3, not '$ROBOT'" >&2; exit 1 ;;
esac
HOST="${1:-${STATION_SSH_HOST:-$DEFAULT_HOST}}"
REPO="${2:-${STATION_REPO:-~/HSA-UMI}}"
FRANKA_WS="${STATION_FRANKA_WS:-$DEFAULT_WS}"
FRANKA_PREFIX="${STATION_FRANKA_PREFIX:-$DEFAULT_PREFIX}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTROLLER=external/franka_streaming_impedance_controller

# An uninitialised submodule is an EMPTY directory here, and --delete would erase the station's
# copy of it. Refuse rather than ship the deletion.
if [ -z "$(ls -A "${HERE}/${CONTROLLER}" 2>/dev/null)" ]; then
    echo "error: ${CONTROLLER} is empty (uninitialised submodule); --delete would wipe it on ${HOST}." >&2
    echo "       run: git submodule update --init ${CONTROLLER}" >&2
    exit 1
fi

echo "==> [robot ${ROBOT}] Copying nuc/ + the controller to ${HOST}:${REPO} (tar over ssh) ..."
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
    test -f ${REPO}/nuc/launch/bringup.launch.py
    test -f ${REPO}/nuc/config/cyclonedds_station.xml
    echo '    nuc/ present, controller built'
"

echo "==> Done. NOTE: controller_manager keeps the old plugin .so mapped, so a rebuilt controller"
echo "    runs only after bringup restarts. Next, on ${HOST} (or via ./hsa_session.sh robot:=${ROBOT}):"
echo "      export HSA_ROBOT=${ROBOT}; source ${REPO}/nuc/station_env.sh"
echo "      ros2 launch ${REPO}/nuc/launch/bringup.launch.py robot:=${ROBOT}"
