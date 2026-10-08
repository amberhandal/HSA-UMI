"""The collect_ui helpers: force pairing by stamp and the overlay text, without a window or ROS traffic."""

import numpy as np

from hsa_ros2 import collect_ui


def test_pair_force_takes_newest_sample_at_or_before_the_frame():
    """Check the pairing rule and the reported age."""
    times = [1.00, 1.01, 1.02]
    w = [np.full(6, i, dtype=float) for i in range(3)]
    f, age = collect_ui.pair_force(times, w, 1.015)
    assert f[0] == 1 and abs(age - 0.005) < 1e-9
    assert collect_ui.pair_force(times, w, 0.5) == (None, None)
    assert collect_ui.pair_force([], [], 1.0) == (None, None)


def test_overlay_lines():
    """Check the four lines: demo and state, force with its age, timestamp, stage."""
    status = {'episode': 3, 'recording': True, 'elapsed_s': 2.5, 'steps': 25, 't0': 100.0}
    lines = collect_ui.overlay_lines(status, np.array([1.0, 2.0, 2.0, 0, 0, 0]), 0.004, 102.25, 1)
    assert lines[0] == 'Demo 3   REC 2.5 s   25 steps'
    assert lines[1].startswith('Force   3.0 N') and lines[1].endswith('sample 4 ms old')
    assert lines[2].endswith('t = 2.250 s')
    assert lines[3] == 'Stage 1'
    idle = collect_ui.overlay_lines({}, None, None, 102.25, 0)
    assert idle[0] == 'Demo ?   idle' and idle[1] == 'Force   no data'


def test_draw_overlay_keeps_shape_and_marks_recording():
    """Check that drawing returns a same-shaped copy and paints the red dot only while recording."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = collect_ui.draw_overlay(img, ['a', 'b'], recording=True)
    assert out.shape == img.shape and img.max() == 0
    assert out[30, 610, 2] == 255 and out[30, 610, 0] == 0
    assert collect_ui.draw_overlay(img, ['a'], recording=False)[30, 610].max() == 0
