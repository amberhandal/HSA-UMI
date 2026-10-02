# shellcheck shell=sh
# handell (Amber's dev laptop): its link to the HSA lab's Franka station. Sourced by
# setup_franka_env.sh when `hostname -s` is handell. `:=` throughout, so an exported override wins.
#
# The station is an NU-MSR Franka workstation (https://nu-msr.github.io): br0 = 192.168.18.1
# leases 192.168.18.0/24, names the laptop from its lease, points the station's chrony at it and
# default-routes the station through it (franka_route.sh). All of that keys off the DHCP LEASE, so
# this laptop must keep taking one -- a manual static IP breaks the hostname, the clock source and
# the route -- and the existing DHCP profile is named below so setup_franka_env.sh reuses it.
# Measured 2026-09-28: 1 Gbit, 0.4 ms RTT to the station.
#
# The cable may be in the built-in port (enp0s31f6) or a USB adapter (enx...) and the address is a
# lease, so both are read off the live routing table at source time rather than pinned.
_hsa_route="$(ip -o route get 192.168.18.1 2>/dev/null)"
_hsa_iface="$(printf '%s' "$_hsa_route" | sed -n 's/.* dev \([^ ]*\).*/\1/p')"
_hsa_self="$(printf '%s' "$_hsa_route" | sed -n 's/.* src \([^ ]*\).*/\1/p')"
if [ -z "$_hsa_iface" ] || printf '%s' "$_hsa_route" | grep -q '^default\|via'; then
  echo "[env.handell] WARNING: no direct route to the station (192.168.18.1) -- is the cable in? Falling back to enp0s31f6." >&2
  _hsa_iface="${_hsa_iface:-enp0s31f6}"; _hsa_self="${_hsa_self:-192.168.18.62}"
fi
: "${FR3_IFACE:=$_hsa_iface}"
: "${FR3_SELF_IP:=${_hsa_self}/24}"
# The DHCP profile owning the live interface, so setup_franka_env.sh finds it "already active".
: "${FR3_NM_PROFILE:=$(nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null | awk -F: -v d="$FR3_IFACE" '$2==d{print $1; exit}')}"
: "${FR3_NM_PROFILE:=Wired connection 1}"
# CycloneDDS: the interface name is baked into the config, so write one for today's interface.
_hsa_dds_dir="${XDG_STATE_HOME:-$HOME/.local/state}/hsa"; mkdir -p "$_hsa_dds_dir"
sed -e "s/enp0s31f6/${FR3_IFACE}/" -e "s#192.168.18.62#${_hsa_self}#" \
  "${POLYUMI_ROOT}/ros2_ws/config/cyclonedds_handell.xml" > "${_hsa_dds_dir}/cyclonedds_handell.xml"
: "${CYCLONEDDS_CONFIG_FILE:=${_hsa_dds_dir}/cyclonedds_handell.xml}"
echo "[env.handell] station link: ${FR3_IFACE} ${_hsa_self} (profile '${FR3_NM_PROFILE}'); DDS config ${CYCLONEDDS_CONFIG_FILE}"
unset _hsa_route _hsa_iface _hsa_self _hsa_dds_dir
# franka_description for Foxglove's URDF panel: this laptop has franka_ros2 v0.1.15 built here.
: "${FRANKA_DESCRIPTION_WS:=${HOME}/ws/franka/install}"

# FastDDS on both sides: the station has no rmw_cyclonedds_cpp (nor internet to install it), and on
# one L2 subnet FastDDS needs no config file. nuc/station_env.sh reads the same variable.
: "${HSA_DDS:=fastdds}"
