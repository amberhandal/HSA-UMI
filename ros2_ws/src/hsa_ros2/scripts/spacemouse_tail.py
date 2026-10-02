#!/usr/bin/env python3
"""
Tell the puck's own settle tail from the reader's backlog when the arm keeps moving after a release.

Run at the desk, no arm, no ROS, with the teleop stopped (it holds the device):

    /usr/bin/python3 ros2_ws/src/hsa_ros2/scripts/spacemouse_tail.py [seconds]

Push and release the puck a few times, holding some pushes for 2-3 s. The device is opened twice (hidraw
allows it): handle A drains every report as it arrives (what the device says, and when it says zero);
handle B is read once per 5 ms like spacemouse_teleop's reader thread (what the node integrates). Per
release it prints the device's settle tail, how long B kept reporting motion after A went quiet, and the
creep each would cause at ``max_pos_speed`` 0.2 m/s. Measured 2026-10-02 on the Compact: 125 reports/s
while deflected, reader lag 0-3 ms, device tail 50-160 ms (2-7 mm, the hand lifting off).
"""

from __future__ import annotations

import sys
import time

import numpy as np

MAX_POS_SPEED = 0.2  # m/s at full deflection, teleop.yaml's max_pos_speed
AXES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
READER_PERIOD = 0.005  # spacemouse_teleop's reader thread: one read, then a 5 ms sleep


def read6(s) -> np.ndarray:
    """Return the six axes of a pyspacemouse state as an array."""
    return np.array([s.x, s.y, s.z, s.roll, s.pitch, s.yaw], dtype=np.float64)


def record(seconds: float):
    """Sample both handles for ``seconds``; returns (ta, va, tb, vb), tb/vb None without handle B."""
    import pyspacemouse

    a = pyspacemouse.open()
    if not a:
        raise SystemExit('pyspacemouse.open() failed: is the puck plugged in, and is the teleop stopped?')
    b = pyspacemouse.open()
    if not b:
        print('second open failed: measuring the device only, not the reader')
    print(f'recording for {seconds:.0f} s: push and release the puck a few times, hold some pushes for 2-3 s')
    rows_a: list[tuple[float, np.ndarray]] = []
    rows_b: list[tuple[float, np.ndarray]] = []
    t0 = time.monotonic()
    last_t = None
    next_b = t0
    while True:
        now = time.monotonic() - t0
        if now > seconds:
            break
        s = a.read()
        if s.t != last_t:
            last_t = s.t
            rows_a.append((now, read6(s)))
        if b and now + t0 >= next_b:
            next_b += READER_PERIOD
            rows_b.append((now, read6(b.read())))
        time.sleep(0.0002)
    a.close()
    if b:
        b.close()
    ta = np.array([r[0] for r in rows_a])
    va = np.array([r[1] for r in rows_a]).reshape(-1, 6)
    if not b:
        return ta, va, None, None
    return ta, va, np.array([r[0] for r in rows_b]), np.array([r[1] for r in rows_b]).reshape(-1, 6)


def releases(ta: np.ndarray, mag_a: np.ndarray) -> list[tuple[int, int]]:
    """Return (first deflected index, first all-zero index) of every push held longer than 0.3 s."""
    out = []
    i = 0
    while i < len(ta):
        if mag_a[i] > 0.2:
            j = i
            while j < len(ta) and mag_a[j] != 0.0:
                j += 1
            if j < len(ta) and ta[j] - ta[i] > 0.3:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def main() -> int:
    """Record, then report per release."""
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    ta, va, tb, vb = record(seconds)
    if len(ta) < 10:
        print(f'only {len(ta)} reports arrived: nothing was pushed?')
        return 1
    mag_a = np.abs(va).max(axis=1)
    moving = mag_a > 0.05
    if moving.sum() > 20:
        dts = np.diff(ta)[moving[1:]]
        print(
            f'device: {len(ta)} reports in {seconds:.0f} s; while deflected one every '
            f'{np.median(dts) * 1e3:.1f} ms = {1 / np.median(dts):.0f} reports/s '
            f'(a reader at one per {READER_PERIOD * 1e3:.0f} ms keeps up with {1 / READER_PERIOD:.0f}/s)'
        )
    rel = releases(ta, mag_a)
    if not rel:
        print('no push-and-release of more than 0.3 s found')
        return 1
    print(f'{len(rel)} releases:')
    print('  axis   held     device settle tail        reader still moving after device zero')
    for i, j in rel:
        axis = AXES[int(np.argmax(np.abs(va[i:j]).max(axis=0)))]
        peak = np.abs(va[i:j]).max()
        k = j - 1
        while k > i and mag_a[k] < 0.5 * peak:
            k -= 1
        tail_t = ta[j] - ta[k]
        tail_mm = float(np.sum(mag_a[k:j] * np.diff(ta[k : j + 1]))) * MAX_POS_SPEED * 1e3
        line = f'  {axis:5s} {ta[j] - ta[i]:4.1f} s   {tail_t * 1e3:5.0f} ms {tail_mm:6.1f} mm creep'
        if tb is not None:
            mag_b = np.abs(vb).max(axis=1)
            quiet = np.where((tb >= ta[j]) & (mag_b == 0.0))[0]
            if len(quiet):
                lag = tb[quiet[0]] - ta[j]
                sel = (tb >= ta[j]) & (tb < tb[quiet[0]])
                creep = float(np.sum(mag_b[sel]) * READER_PERIOD) * MAX_POS_SPEED * 1e3
                line += f'    {lag * 1e3:5.0f} ms {creep:6.1f} mm creep'
            else:
                line += '    never went quiet'
        print(line)
    print(
        'creep = reading x time x 0.2 m/s: what the target integrates after the hand lets go.\n'
        'a reader lag that grows with the hold is a backlog (fixed by draining the device);\n'
        'a device tail of more than a few mm is the puck itself (raise spacemouse.deadzone).'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
