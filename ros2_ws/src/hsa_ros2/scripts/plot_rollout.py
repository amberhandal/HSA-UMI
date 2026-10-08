#!/usr/bin/env python3
"""
Figures for one rollout bag: chunks against the executed path, latency timeline, tracking and force.

    scripts/record_rollout.sh peg_01                       # while the policy (or teleop) runs
    scripts/plot_rollout.py ~/hsa_data/rollouts/peg_01     # -> <bag>/figures/*.png

Needs the ROS environment sourced (it reads the bag with rosbag2 and replays TF).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from hsa_ros2 import figures
from hsa_ros2.bag_reader import Rollout


def main() -> int:
    """Run the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('bag', help='bag directory written by record_rollout.sh')
    ap.add_argument('--out', help='output directory (default: <bag>/figures)')
    ap.add_argument('--base', default='fer_link0')
    ap.add_argument('--tcp', default='hsa_tcp')
    args = ap.parse_args()
    roll = Rollout(args.bag, base_frame=args.base, tcp_frame=args.tcp)
    print(roll.summary())
    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(args.bag) / 'figures'
    name = pathlib.Path(args.bag).name
    written = []
    if roll.chunks or len(roll.executed_times):
        written.append(
            figures.plot_chunks_vs_executed(
                roll.chunks, roll.executed_times, roll.executed_xyz, out_dir / 'chunks_vs_executed.png', title=name
            )
        )
    if roll.diag:
        written.append(figures.plot_latency_timeline(roll.diag, out_dir / 'latency_timeline.png', title=name))
    t, err = roll.tracking_error()
    if len(t) or roll.wrench_times:
        written.append(
            figures.plot_tracking_force(
                t, err, roll.wrench_times, roll.wrench_norm, out_dir / 'tracking_force.png', title=name
            )
        )
    for w in written:
        print(w)
    if not written:
        print('nothing to plot: the bag has no chunks, TF, diagnostics or wrench')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
