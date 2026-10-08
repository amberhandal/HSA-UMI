#!/usr/bin/env python3
"""
Tap test: check that the force stream and the frames of a recorded episode are in sync.

    scripts/check_sync.py ~/hsa_data/session --episode 0     # -> <session>/figures/sync_ep0000.png

Record a short episode with the wrench enabled, tap the tool once where the camera can see it,
stop. This finds the force spike, prints the age of the wrench sample paired with each frame (from
the episode's sidecar) and draws the frames around the spike: the tap should be visible in the
outlined frame or its neighbour. If it is consistently one frame later, raise latency.camera.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from hsa_ros2 import figures
from hsa_ros2.dataset_writer import KEY_IMAGE, KEY_TIME, KEY_WRENCH, open_replay_buffer


def main() -> int:
    """Run the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('session', help='session directory written by demo_recorder')
    ap.add_argument('--episode', type=int, default=-1, help='episode index (default: the last)')
    ap.add_argument('--out', help='output directory (default: <session>/figures)')
    args = ap.parse_args()
    session = pathlib.Path(args.session).expanduser()
    root = open_replay_buffer(session)
    if KEY_WRENCH not in root['data']:
        print('the dataset has no wrench stream: record with wrench.enabled:=true')
        return 1
    ends = root['meta']['episode_ends'][:]
    slices = figures.episode_slices(ends)
    k = args.episode if args.episode >= 0 else len(slices) - 1
    a, b = slices[k]
    force = np.linalg.norm(root['data'][KEY_WRENCH][a:b, :3], axis=1)
    times = root['data'][KEY_TIME][a:b].astype(np.float64)
    spike = figures.find_force_spike(times, force)
    sidecar = session / 'episodes' / f'episode_{k:04d}.json'
    if sidecar.exists():
        stats = json.loads(sidecar.read_text()).get('wrench_sample_age_s')
        if stats:
            med = stats.get('median')
            mx = stats.get('max')
            print(
                f'wrench sample age at each frame: median {med * 1e3 if med is not None else float("nan"):.1f} ms, '
                f'max {mx * 1e3 if mx is not None else float("nan"):.1f} ms, '
                f'{stats.get("missing_steps", 0)} frames without a sample'
            )
    print(
        f'episode {k}: {b - a} frames; force spike of {np.diff(force)[spike - 1] if spike else 0:.1f} N '
        f'at frame {spike}, t = {times[spike] - times[0]:.3f} s'
    )
    frames = root['data'][KEY_IMAGE][a:b]
    out_dir = pathlib.Path(args.out) if args.out else session / 'figures'
    out = figures.plot_sync_sheet(
        frames, times, force, spike, out_dir / f'sync_ep{k:04d}.png', title=f'{session.name} episode {k}'
    )
    print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
