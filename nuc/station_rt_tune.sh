#!/usr/bin/env bash
# Real-time hygiene for the station's 1 kHz Franka loop. Run ON THE STATION, as root:
#
#     sudo bash ~/HSA-UMI/nuc/station_rt_tune.sh          # apply (runtime, reversible, no reboot)
#     sudo bash ~/HSA-UMI/nuc/station_rt_tune.sh status   # just show the current state
#     sudo bash ~/HSA-UMI/nuc/station_rt_tune.sh revert   # powersave governor back, release the idle hold
#
# Why: on the `powersave` governor with deep idle states allowed, a late wake-up or a frequency ramp
# is longer than the 1 ms the FCI allows; the robot's command success rate then dips and the arm's
# motion is visibly rougher. Franka's control-PC guidance is the same: performance governor, no CPU
# power saving.
#
# At runtime: the performance governor on every core, and a holder process keeping /dev/cpu_dma_latency
# at 0 us (the kernel then keeps the cores out of deep idle states) for as long as it runs. What it
# cannot do at runtime is the memlock limit, read at login from /etc/security/limits.conf: a 100 MB cap
# stops ros2_control_node pinning its memory ("Unable to lock the memory"). Change the two lines
#     @realtime soft memlock 102400
#     @realtime hard memlock 102400
# to `unlimited` and log in again.
set -euo pipefail
HOLD_PID_FILE=/run/hsa_cpu_dma_latency.pid
mode=${1:-apply}

status() {
  echo "governor: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  echo "intel_idle max_cstate: $(cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null || echo n/a)"
  if [ -f "$HOLD_PID_FILE" ] && kill -0 "$(cat "$HOLD_PID_FILE")" 2>/dev/null; then echo "cpu_dma_latency hold: active (pid $(cat "$HOLD_PID_FILE"))"; else echo "cpu_dma_latency hold: none"; fi
  echo "memlock limit (this shell): $(ulimit -l) KB  -- 'unlimited' needed for ros2_control_node to pin memory"
}

case "$mode" in
  status) status ;;
  apply)
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done
    if ! { [ -f "$HOLD_PID_FILE" ] && kill -0 "$(cat "$HOLD_PID_FILE")" 2>/dev/null; }; then
      # A process that holds /dev/cpu_dma_latency open with 0 written keeps the cores out of deep
      # idle states until it exits (the kernel honours the request only while the fd is open).
      nohup python3 -c "import time; f = open('/dev/cpu_dma_latency', 'wb'); f.write(b'\x00\x00\x00\x00'); f.flush(); time.sleep(10**9)" >/dev/null 2>&1 &
      echo $! > "$HOLD_PID_FILE"
    fi
    status ;;
  revert)
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo powersave > "$g"; done
    [ -f "$HOLD_PID_FILE" ] && { kill "$(cat "$HOLD_PID_FILE")" 2>/dev/null || true; rm -f "$HOLD_PID_FILE"; }
    status ;;
  *) echo "usage: $0 [apply|status|revert]"; exit 2 ;;
esac
