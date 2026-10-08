#!/usr/bin/env python3
"""
Workspace coverage of a recorded dataset: every episode's TCP path from above and the side.

    scripts/plot_dataset.py ~/hsa_data/session            # a session dir, its replay_buffer.zarr, or a .zarr.zip
    scripts/plot_dataset.py ~/hsa_data/session --3d --box config/teleop.yaml

Writes workspace_coverage.png next to the data (or into --out).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

from hsa_ros2 import figures
from hsa_ros2.dataset_writer import KEY_POS, open_replay_buffer


def main() -> int:
    """Run the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('path')
    ap.add_argument('--out', help='output directory (default: next to the data)')
    ap.add_argument('--3d', dest='three_d', action='store_true', help='add a 3-D view of the paths')
    ap.add_argument('--box', help="teleop.yaml whose workspace box to draw, or 'x_min,x_max,y_min,y_max,z_min,z_max'")
    args = ap.parse_args()
    root = open_replay_buffer(args.path)
    pos = root['data'][KEY_POS][:]
    ends = root['meta']['episode_ends'][:]
    box = None
    if args.box:
        if pathlib.Path(args.box).exists():
            box = yaml.safe_load(open(args.box))['/**']['ros__parameters']['workspace']
        else:
            vals = [float(v) for v in args.box.split(',')]
            box = dict(zip(('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'), vals))
    src = pathlib.Path(args.path)
    out_dir = pathlib.Path(args.out) if args.out else (src if src.is_dir() else src.parent) / 'figures'
    out = figures.plot_workspace(
        pos, ends, out_dir / 'workspace_coverage.png', box=box, three_d=args.three_d, title=src.name
    )
    print(f'{len(ends)} episodes, {len(pos)} steps -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
