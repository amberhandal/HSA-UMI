#!/usr/bin/env python3
"""
Recorded external wrench against the demo phases: is the force real contact or motion artifact?

    scripts/plot_wrench_phases.py data/key_in_hole/rigid/v1/key_in_hole_rigid_v1.zarr.zip
    scripts/plot_wrench_phases.py data/key_in_hole/rigid/v1 --near-cm 2 --examples 0,10,20

Phases come from the recorded subtask labels, with the insert stage split by distance to the pose
where it ended (the key in the hole, where the operator pressed turn): approach (> --near-cm away,
free-space motion) and near hole. The FER's wrench is estimated from joint torques, so force during
the approach that tracks TCP speed is the estimate's motion error, not contact.

Writes wrench_phases.png (examples over time, per-phase distributions, approach force vs speed)
next to the data (or into --out) and prints a per-phase table.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hsa_ros2.dataset_writer import KEY_POS, open_replay_buffer  # noqa: E402

PHASES = ('approach', 'near hole', 'turn', 'open')
COLORS = ('#4c78a8', '#f58518', '#54a24b', '#b279a2')
AXES = ('Fx', 'Fy', 'Fz')


def phase_labels(pos: np.ndarray, stage: np.ndarray, near_m: float) -> np.ndarray:
    """Per-step phase index (into PHASES) for one episode."""
    labels = np.where(stage >= 1, stage + 1, 0).astype(int)
    insert = np.flatnonzero(stage == 0)
    if len(insert):
        hole = pos[insert[-1]]
        near = np.linalg.norm(pos[insert] - hole, axis=1) <= near_m
        labels[insert[near]] = 1
    return labels


def main() -> int:
    """Run the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('path', help='a .zarr.zip, a session dir, or its replay_buffer.zarr')
    ap.add_argument('--out', help='output directory (default: next to the data)')
    ap.add_argument('--near-cm', type=float, default=2.0, help='insert steps within this of the hole are "near hole"')
    ap.add_argument('--examples', default='0,1,2', help='comma-separated episode indices to plot over time')
    args = ap.parse_args()

    root = open_replay_buffer(args.path)
    data = root['data']
    pos = data[KEY_POS][:]
    wrench = data['robot0_ext_wrench'][:]
    stage = data['stage'][:].reshape(-1)
    t = data['timestamp'][:].reshape(-1)
    ends = root['meta']['episode_ends'][:]
    starts = np.r_[0, ends[:-1]]

    labels = np.empty(len(pos), dtype=int)
    speed = np.zeros(len(pos))
    for a, b in zip(starts, ends):
        labels[a:b] = phase_labels(pos[a:b], stage[a:b], args.near_cm / 100)
        if b - a > 1:
            v = np.linalg.norm(np.diff(pos[a:b], axis=0), axis=1) / np.maximum(np.diff(t[a:b]), 1e-3)
            speed[a:b] = np.r_[v[0], v]
    fmag = np.linalg.norm(wrench[:, :3], axis=1)

    print(f'{len(ends)} episodes, {len(pos)} steps; near hole = insert steps within {args.near_cm:g} cm of its end')
    print(f'{"phase":<10} {"frames":>7} {"share":>6}  ' + '  '.join(f'{a + " med":>8}' for a in AXES)
          + f'  {"|F| med":>8} {"|F| p95":>8} {"speed mm/s":>10}')
    for i, name in enumerate(PHASES):
        m = labels == i
        if not m.any():
            continue
        meds = '  '.join(f'{np.median(wrench[m, k]):8.1f}' for k in range(3))
        print(f'{name:<10} {m.sum():7d} {m.mean():6.0%}  {meds}  {np.median(fmag[m]):8.1f} '
              f'{np.percentile(fmag[m], 95):8.1f} {1000 * np.median(speed[m]):10.0f}')
    appr = labels == 0
    r = np.corrcoef(speed[appr], fmag[appr])[0, 1] if appr.sum() > 2 else float('nan')
    print(f'approach: corr(|F|, TCP speed) = {r:.2f}  (near 0: force not driven by motion; large: motion artifact)')

    examples = [int(e) for e in args.examples.split(',') if e.strip() != '' and int(e) < len(ends)]
    fig = plt.figure(figsize=(15, 3.2 * len(examples) + 4.5), constrained_layout=True)
    grid = fig.add_gridspec(len(examples) + 1, 3)
    for row, ep in enumerate(examples):
        a, b = starts[ep], ends[ep]
        ax = fig.add_subplot(grid[row, :])
        tt = t[a:b] - t[a]
        for i in range(len(PHASES)):
            m = labels[a:b] == i
            if m.any():
                ax.fill_between(tt, 0, 1, where=m, transform=ax.get_xaxis_transform(), color=COLORS[i],
                                alpha=0.12, step='mid', label=PHASES[i] if row == 0 else None)
        for k, ls in zip(range(3), ('-', '--', ':')):
            ax.plot(tt, wrench[a:b, k], 'k', ls=ls, lw=1, label=AXES[k] if row == 0 else None)
        ax.set_ylabel('force (N)')
        ax.set_title(f'episode {ep}')
        sp = ax.twinx()
        sp.plot(tt, 1000 * speed[a:b], color='#e45756', lw=0.8, alpha=0.7)
        sp.set_ylabel('TCP speed (mm/s)', color='#e45756')
        if row == 0:
            ax.legend(loc='upper right', ncol=7, fontsize=8)
    ax.set_xlabel('time in episode (s)')

    ax = fig.add_subplot(grid[-1, :2])
    groups = [(k, i) for k in range(4) for i in range(len(PHASES))]
    vals = [(fmag if k == 3 else wrench[:, k])[labels == i] for k, i in groups]
    keep = [j for j, v in enumerate(vals) if len(v)]
    bp = ax.boxplot([vals[j] for j in keep], positions=[groups[j][0] * 5 + groups[j][1] for j in keep],
                    widths=0.8, showfliers=False, patch_artist=True)
    for patch, j in zip(bp['boxes'], keep):
        patch.set_facecolor(COLORS[groups[j][1]])
    ax.set_xticks([k * 5 + 1.5 for k in range(4)], [*AXES, '|F|'])
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_ylabel('force (N)')
    ax.set_title('force per phase (box = 25–75 %, whiskers = 1.5 IQR)')

    ax = fig.add_subplot(grid[-1, 2])
    ax.scatter(1000 * speed[appr], fmag[appr], s=2, alpha=0.3, color=COLORS[0])
    ax.set_xlabel('TCP speed (mm/s)')
    ax.set_ylabel('|F| (N)')
    ax.set_title(f'approach (free space): |F| vs speed, r = {r:.2f}')

    src = pathlib.Path(args.path)
    out_dir = pathlib.Path(args.out) if args.out else (src if src.is_dir() else src.parent) / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'wrench_phases.png'
    fig.savefig(out, dpi=110)
    print(f'-> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
