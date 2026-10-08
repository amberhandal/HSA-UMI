"""Each figure draws from synthetic arrays and writes a PNG; the series helpers are checked exactly."""

import numpy as np

from hsa_ros2 import figures


def _episodes(n_ep=3, n_steps=40, seed=0):
    rng = np.random.default_rng(seed)
    pos = []
    for k in range(n_ep):
        t = np.linspace(0, 1, n_steps)
        pos.append(np.stack([0.3 + 0.2 * t + 0.01 * k, 0.1 * np.sin(2 * np.pi * t) + 0.05 * k, 0.3 - 0.1 * t], 1))
    pos = np.concatenate(pos) + rng.normal(0, 1e-3, (n_ep * n_steps, 3))
    ends = np.arange(1, n_ep + 1) * n_steps
    return pos, ends


def test_episode_slices():
    """Check that the slices tile the step range."""
    assert figures.episode_slices([10, 25, 30]) == [(0, 10), (10, 25), (25, 30)]
    assert figures.episode_slices([]) == []


def test_commanded_at_prefers_newest_chunk_and_holds_past_its_end():
    """Check the chunk selection, interpolation and hold."""
    c1 = (np.array([1.0, 2.0]), np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    c2 = (np.array([1.5, 2.5]), np.array([[5.0, 0, 0], [6.0, 0, 0]]))
    assert figures.commanded_at([c1, c2], 0.5) is None
    assert np.allclose(figures.commanded_at([c1, c2], 1.25), [0.25, 0, 0])
    assert np.allclose(figures.commanded_at([c1, c2], 2.0), [5.5, 0, 0])
    assert np.allclose(figures.commanded_at([c1, c2], 9.0), [6.0, 0, 0])


def test_find_force_spike_and_wilson():
    """Check the spike index and the interval's bounds."""
    f = np.array([1.0, 1.1, 1.0, 8.0, 7.5, 1.2])
    assert figures.find_force_spike(np.arange(6), f) == 3
    lo, hi = figures.wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.93 < hi < 0.96
    assert figures.wilson(0, 0) == (0.0, 0.0)


def test_workspace_figure(tmp_path):
    """Check that the coverage figure is written, with and without the 3-D view and box."""
    pos, ends = _episodes()
    box = dict(x_min=0.25, x_max=0.72, y_min=-0.35, y_max=0.35, z_min=0.05, z_max=0.5)
    out = figures.plot_workspace(pos, ends, tmp_path / 'ws.png', box=box, title='t')
    out3 = figures.plot_workspace(pos, ends, tmp_path / 'ws3.png', three_d=True)
    assert out.stat().st_size > 1000 and out3.stat().st_size > 1000


def test_rollout_figures(tmp_path):
    """Check the chunk, latency and tracking figures on synthetic series."""
    t = np.arange(0, 5, 0.02)
    executed = np.stack([0.4 + 0.05 * np.sin(t), 0.1 * np.cos(t), 0.3 + 0 * t], 1)
    chunks = [
        (
            t0 + 0.1 * np.arange(1, 17),
            np.stack(
                [
                    0.4 + 0.05 * np.sin(t0 + 0.1 * np.arange(1, 17)),
                    0.1 * np.cos(t0 + 0.1 * np.arange(1, 17)),
                    0.3 + np.zeros(16),
                ],
                1,
            ),
        )
        for t0 in np.arange(0, 4, 0.8)
    ]
    out = figures.plot_chunks_vs_executed(chunks, t, executed, tmp_path / 'c.png', title='t')
    series = {
        'obs_age_s': (np.arange(0, 5, 0.8), np.full(7, 0.05)),
        'inference_latency_s': (np.arange(0, 5, 0.8), np.linspace(0.2, 0.25, 7)),
        'n_published_arm': (np.arange(0, 5, 0.8), np.full(7, 8)),
    }
    out2 = figures.plot_latency_timeline(series, tmp_path / 'l.png')
    out3 = figures.plot_tracking_force(t, np.abs(np.sin(t)) * 0.01, t, 2 + np.abs(np.cos(t)), tmp_path / 'tf.png')
    assert all(o.stat().st_size > 1000 for o in (out, out2, out3))


def test_outcomes_figure(tmp_path):
    """Check the ledger figure and the returned rate."""
    rows = [
        dict(rollout='1', dx_mm='0', dy_mm='0', outcome='success', failure_code='', insertion_depth_mm='25'),
        dict(rollout='2', dx_mm='20', dy_mm='0', outcome='failure', failure_code='miss', insertion_depth_mm='0'),
        dict(rollout='3', dx_mm='0', dy_mm='20', outcome='success', failure_code='', insertion_depth_mm='24'),
    ]
    out, (rate, lo, hi) = figures.plot_outcomes(rows, tmp_path / 'o.png')
    assert out.stat().st_size > 1000 and abs(rate - 2 / 3) < 1e-9 and lo < rate < hi


def test_sync_sheet(tmp_path):
    """Check the tap-test sheet on synthetic frames with a spike in the middle."""
    n = 20
    frames = np.random.default_rng(0).integers(0, 255, (n, 48, 64, 3), dtype=np.uint8)
    times = np.arange(n) / 30.0
    force = np.ones(n)
    force[12:15] = 9.0
    out = figures.plot_sync_sheet(frames, times, force, figures.find_force_spike(times, force), tmp_path / 's.png')
    assert out.stat().st_size > 1000
