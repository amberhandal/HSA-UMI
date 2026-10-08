#!/usr/bin/env python3
"""
The evaluation ledger as a figure: start offsets coloured by outcome, insertion depth per rollout.

    scripts/plot_outcomes.py ~/hsa_data/eval/ledger.csv       # -> outcomes.png next to it

The ledger's columns are those of config/eval_ledger_example.csv; copy it to start a new one.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

from hsa_ros2 import figures


def main() -> int:
    """Run the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('ledger', help='CSV with the columns of config/eval_ledger_example.csv')
    ap.add_argument('--out', help='output directory (default: next to the ledger)')
    args = ap.parse_args()
    path = pathlib.Path(args.ledger)
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if any(v.strip() for v in r.values() if v)]
    if not rows:
        print('empty ledger')
        return 1
    out_dir = pathlib.Path(args.out) if args.out else path.parent
    out, (rate, lo, hi) = figures.plot_outcomes(rows, out_dir / 'outcomes.png', title=path.stem)
    print(f'{len(rows)} rollouts, success {rate * 100:.0f}% (95% Wilson {lo * 100:.0f} to {hi * 100:.0f}%) -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
