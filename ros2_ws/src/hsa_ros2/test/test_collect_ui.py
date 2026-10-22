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


def test_wrist_line_from_recorder_status():
    """The fifth line appears only when the recorder reports wrist streams, and says what is missing."""
    assert collect_ui.wrist_line({}) is None
    assert len(collect_ui.overlay_lines({}, None, None, 1.0, 0)) == 4
    status = {
        'wrist_ext_deg': [360.0, 359.6, 360.2, 358.9, 360.0, 361.0],
        'wrist_iq_a': [1.9, -2.1, 1.8, 2.0, 1.7, 1.9],
        'wrist_errors': [0] * 6,
        'plate_pos_mm': [0.1, -0.2, 12.3],
        'plate_tilt_deg': 1.4,
        'plate_visible': True,
        'wrist_camera_age_ms': 8.0,
    }
    line = collect_ui.wrist_line(status)
    assert line.startswith('Wrist 360 360 360 359 360 361 deg   Iq 2.1 A')
    assert 'plate z +12.3 mm tilt 1.4 deg seen' in line and line.endswith('cam1 8 ms')
    assert len(collect_ui.overlay_lines(status, None, None, 1.0, 0)) == 5
    lost = collect_ui.wrist_line({'wrist_ext_deg': None, 'plate_pos_mm': None, 'wrist_camera_age_ms': None})
    assert lost == 'Wrist   no data   plate: no pose (zero it)   cam1 no frame'
    assert 'ERRORS' in collect_ui.wrist_line({'wrist_ext_deg': [0] * 6, 'wrist_errors': [0, 8, 0, 0, 0, 0]})
