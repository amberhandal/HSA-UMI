"""
The project's standard figures, drawn from plain arrays so the scripts and the tests share them.

Workspace coverage of a dataset, predicted chunks against the executed path, the client's latency
timeline, tracking error and force over a rollout, the evaluation ledger, and the tap-test sheet
that checks the force stream is in sync with the frames. Every function writes a PNG and returns
its path. Colours: one fixed categorical order, one blue ramp for progression, green/red only for
success/failure.
"""

from __future__ import annotations

import math
import pathlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUE_RAMP = ['#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
GOOD, CRITICAL = '#0ca30c', '#d03b3b'
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
AXES = ('x', 'y', 'z')


def _style(ax, xlabel: str = '', ylabel: str = '') -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, out_path) -> pathlib.Path:
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


def ramp_color(i: int, n: int) -> str:
    """Colour for item ``i`` of ``n`` along the blue ramp (light first)."""
    if n <= 1:
        return BLUE_RAMP[-1]
    return BLUE_RAMP[min(len(BLUE_RAMP) - 1, int(i * len(BLUE_RAMP) / n))]


# ----------------------------------------------------------------------------------------------
# Series helpers
# ----------------------------------------------------------------------------------------------


def episode_slices(episode_ends) -> list[tuple[int, int]]:
    """``[(start, end), ...]`` step ranges from the UMI ``meta/episode_ends`` array."""
    ends = [int(e) for e in np.asarray(episode_ends).ravel()]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def commanded_at(chunks: list[tuple[np.ndarray, np.ndarray]], t: float) -> np.ndarray | None:
    """
    Return the position the controller was following at ``t``.

    That is the newest chunk whose first waypoint is due by then, interpolated linearly between its
    waypoints and held past its last one.
    """
    best = None
    for times, xyz in chunks:
        if times.size and times[0] <= t and (best is None or times[0] >= best[0][0]):
            best = (times, xyz)
    if best is None:
        return None
    times, xyz = best
    return np.array([np.interp(t, times, xyz[:, k]) for k in range(3)])


def find_force_spike(times, force_norm) -> int:
    """Index of the largest one-step rise in ``force_norm``: where a tap landed."""
    f = np.asarray(force_norm, dtype=np.float64)
    if f.size < 2:
        return 0
    return int(np.nanargmax(np.diff(f))) + 1


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success rate."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------


def plot_workspace(pos, episode_ends, out_path, *, box: dict | None = None, three_d: bool = False, title: str = ''):
    """
    Workspace coverage: every episode's TCP path from above and from the side, starts and ends marked.

    :param pos: (N, 3) TCP positions in the base frame, metres, all episodes concatenated.
    :param box: optional ``{x_min, x_max, y_min, y_max, z_min, z_max}`` drawn as the teleop's clamp.
    :param three_d: also draw a 3-D view of the paths.
    """
    pos = np.asarray(pos, dtype=np.float64)
    slices = episode_slices(episode_ends)
    n = len(slices)
    fig, axes = plt.subplots(1, 3 if three_d else 2, figsize=(13 if three_d else 9, 4.2))
    axes = list(np.atleast_1d(axes))
    if three_d:
        axes[2].remove()
        axes[2] = fig.add_subplot(1, 3, 3, projection='3d')
    views = [(0, 1, 'from above', 'x (m)', 'y (m)'), (0, 2, 'from the side', 'x (m)', 'z (m)')]
    for ax, (i, j, name, xl, yl) in zip(axes[:2], views):
        for k, (a, b) in enumerate(slices):
            seg = pos[a:b]
            if len(seg) == 0:
                continue
            ax.plot(seg[:, i], seg[:, j], color=ramp_color(k, n), linewidth=1.0, alpha=0.9)
        starts = np.array([pos[a] for a, b in slices if b > a])
        ends = np.array([pos[b - 1] for a, b in slices if b > a])
        if len(starts):
            ax.scatter(
                starts[:, i],
                starts[:, j],
                s=28,
                facecolors='white',
                edgecolors=INK,
                linewidths=1.2,
                zorder=3,
                label='start',
            )
            ax.scatter(ends[:, i], ends[:, j], s=34, marker='x', color=INK, linewidths=1.2, zorder=3, label='end')
        if box:
            lo = (box[f'{AXES[i]}_min'], box[f'{AXES[j]}_min'])
            hi = (box[f'{AXES[i]}_max'], box[f'{AXES[j]}_max'])
            ax.add_patch(plt.Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fill=False, linestyle='--', edgecolor=MUTED))
        ax.set_aspect('equal')
        ax.set_title(f'{name}, {n} episodes', fontsize=10, color=INK)
        _style(ax, xl, yl)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    if three_d:
        ax = axes[2]
        for k, (a, b) in enumerate(slices):
            seg = pos[a:b]
            if len(seg):
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=ramp_color(k, n), linewidth=0.9)
        ax.set_xlabel('x (m)', color=MUTED)
        ax.set_ylabel('y (m)', color=MUTED)
        ax.set_zlabel('z (m)', color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_title('paths', fontsize=10, color=INK)
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    return _save(fig, out_path)


def plot_chunks_vs_executed(chunks, executed_times, executed_xyz, out_path, *, title: str = ''):
    """
    Predicted chunks against the executed TCP path: one panel per axis over time, and the top view.

    :param chunks: ``[(times (K,), xyz (K, 3)), ...]`` in absolute seconds.
    :param executed_times: (N,) absolute seconds; ``executed_xyz`` (N, 3).
    """
    et = np.asarray(executed_times, dtype=np.float64)
    ex = np.asarray(executed_xyz, dtype=np.float64)
    t0 = min([et[0]] + [c[0][0] for c in chunks if c[0].size]) if et.size else 0.0
    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
    for k in range(3):
        ax = fig.add_subplot(grid[k, 0])
        for ci, (ct, cx) in enumerate(chunks):
            if ct.size == 0:
                continue
            ax.plot(
                ct - t0,
                cx[:, k],
                color=SERIES[0],
                linewidth=0.8,
                alpha=0.7,
                label='predicted chunk' if ci == 0 else None,
            )
            ax.plot(ct[0] - t0, cx[0, k], marker='o', markersize=2.5, color=SERIES[0])
        if et.size:
            ax.plot(et - t0, ex[:, k], color=INK, linewidth=1.6, label='executed')
        _style(ax, 'time (s)' if k == 2 else '', f'{AXES[k]} (m)')
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax = fig.add_subplot(grid[:, 1])
    for ct, cx in chunks:
        if ct.size:
            ax.plot(cx[:, 0], cx[:, 1], color=SERIES[0], linewidth=0.7, alpha=0.6)
    if et.size:
        ax.plot(ex[:, 0], ex[:, 1], color=INK, linewidth=1.6)
        ax.plot(ex[0, 0], ex[0, 1], marker='o', markersize=6, markerfacecolor='white', markeredgecolor=INK)
    ax.set_aspect('equal')
    ax.set_title('from above', fontsize=10, color=INK)
    _style(ax, 'x (m)', 'y (m)')
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    return _save(fig, out_path)


LATENCY_METRICS = ('obs_age_s', 'image_age_s', 'inference_latency_s', 'inference_model_s')
COUNT_METRICS = ('n_published_arm', 'n_stale_arm')


def plot_latency_timeline(series: dict, out_path, *, title: str = ''):
    """
    Draw the client's diagnostics per inference cycle: ages and latencies (ms) above, action counts below.

    :param series: ``{metric: (times (N,), values (N,))}`` from the ``/polyumi/diag/*`` topics.
    """
    present = [m for m in LATENCY_METRICS if m in series and len(series[m][0])]
    counts = [m for m in COUNT_METRICS if m in series and len(series[m][0])]
    t0 = min(np.asarray(series[m][0])[0] for m in present + counts) if present + counts else 0.0
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    for i, m in enumerate(present):
        t, v = (np.asarray(a, dtype=np.float64) for a in series[m])
        top.plot(t - t0, v * 1e3, color=SERIES[i], linewidth=1.2, marker='.', markersize=3, label=m.replace('_s', ''))
    _style(top, '', 'ms')
    top.set_title('observation age and inference latency per cycle', fontsize=10, color=INK)
    if present:
        top.legend(frameon=False, fontsize=8, ncol=len(present))
    for i, m in enumerate(counts):
        t, v = (np.asarray(a, dtype=np.float64) for a in series[m])
        bottom.step(t - t0, v, where='post', color=SERIES[i], linewidth=1.2, label=m)
    _style(bottom, 'time (s)', 'actions')
    if counts:
        bottom.legend(frameon=False, fontsize=8, ncol=len(counts))
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    return _save(fig, out_path)


def plot_tracking_force(times, error_m, force_times, force_n, out_path, *, title: str = ''):
    """Tracking error (mm) and external force magnitude (N) over a rollout, two panels, one time axis."""
    t = np.asarray(times, dtype=np.float64)
    ft = np.asarray(force_times, dtype=np.float64)
    t0 = min([t[0]] if t.size else [] + ([ft[0]] if ft.size else [])) if (t.size or ft.size) else 0.0
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    if t.size:
        top.plot(t - t0, np.asarray(error_m) * 1e3, color=SERIES[0], linewidth=1.2)
    _style(top, '', 'tracking error (mm)')
    if ft.size:
        bottom.plot(ft - t0, force_n, color=SERIES[1], linewidth=1.2)
    _style(bottom, 'time (s)', 'external force (N)')
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    return _save(fig, out_path)


def plot_outcomes(rows: list[dict], out_path, *, title: str = ''):
    """
    Draw the evaluation ledger: start offsets coloured by outcome, and insertion depth per rollout.

    ``rows`` are the ledger's CSV rows (``dx_mm``, ``dy_mm``, ``outcome``, ``failure_code``,
    ``insertion_depth_mm``, ``rollout``). Returns the path and the success rate with its Wilson interval.
    """
    ok = [r for r in rows if str(r.get('outcome', '')).strip().lower() == 'success']
    bad = [r for r in rows if r not in ok]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1, 1.4]})
    for group, color, marker, label in ((ok, GOOD, 'o', 'success'), (bad, CRITICAL, 'x', 'failure')):
        if group:
            left.scatter(
                [float(r['dx_mm']) for r in group],
                [float(r['dy_mm']) for r in group],
                s=40,
                color=color,
                marker=marker,
                label=label,
                zorder=3,
            )
    for r in bad:
        code = str(r.get('failure_code', '')).strip()
        if code:
            left.annotate(
                code,
                (float(r['dx_mm']), float(r['dy_mm'])),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=7,
                color=MUTED,
            )
    left.set_aspect('equal')
    left.axhline(0, color=GRID, linewidth=0.8)
    left.axvline(0, color=GRID, linewidth=0.8)
    _style(left, 'start offset x (mm)', 'start offset y (mm)')
    left.legend(frameon=False, fontsize=8)
    ids = [str(r.get('rollout', i + 1)) for i, r in enumerate(rows)]
    depths = [float(r.get('insertion_depth_mm', 0) or 0) for r in rows]
    colors = [GOOD if r in ok else CRITICAL for r in rows]
    right.bar(ids, depths, color=colors, width=0.7)
    _style(right, 'rollout', 'insertion depth (mm)')
    lo, hi = wilson(len(ok), len(rows))
    rate = len(ok) / len(rows) if rows else 0.0
    right.set_title(
        f'{len(ok)}/{len(rows)} succeeded: {rate * 100:.0f}% (95% CI {lo * 100:.0f} to {hi * 100:.0f}%)',
        fontsize=10,
        color=INK,
    )
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    return _save(fig, out_path), (rate, lo, hi)


def plot_sync_sheet(frames, frame_times, force_norm, spike_idx: int, out_path, *, n_frames: int = 8, title: str = ''):
    """
    Tap test: the frames around the force spike above the force trace, the spike frame outlined.

    In a synced recording the frame where the tap is visible is the outlined one (or its neighbour,
    at 30 Hz one frame is 33 ms). ``frames`` (N, H, W, 3) RGB and ``force_norm`` (N,) share the
    index with ``frame_times`` (N,) seconds.
    """
    frames = np.asarray(frames)
    ft = np.asarray(frame_times, dtype=np.float64)
    f = np.asarray(force_norm, dtype=np.float64)
    lo = max(0, spike_idx - n_frames // 2)
    hi = min(len(frames), lo + n_frames)
    idx = list(range(lo, hi))
    fig = plt.figure(figsize=(1.6 * len(idx) + 1, 4.8))
    grid = fig.add_gridspec(2, len(idx), height_ratios=[1.1, 1])
    for col, i in enumerate(idx):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frames[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{(ft[i] - ft[spike_idx]) * 1e3:+.0f} ms', fontsize=8, color=INK if i == spike_idx else MUTED)
        for side in ax.spines.values():
            side.set_edgecolor(CRITICAL if i == spike_idx else GRID)
            side.set_linewidth(2.5 if i == spike_idx else 0.8)
    ax = fig.add_subplot(grid[1, :])
    win = slice(max(0, lo - 10), min(len(f), hi + 10))
    ax.plot(ft[win] - ft[spike_idx], f[win], color=SERIES[1], linewidth=1.4, marker='.', markersize=3)
    ax.axvline(0.0, color=CRITICAL, linewidth=1.0)
    for i in idx:
        ax.axvline(ft[i] - ft[spike_idx], color=GRID, linewidth=0.6)
    _style(ax, 'time from the force spike (s)', 'external force (N)')
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
    return _save(fig, out_path)
