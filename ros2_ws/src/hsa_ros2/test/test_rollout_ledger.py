"""The rollout ledger's pure parts: the trial clock, the CSV, the placement and the status line."""

import numpy as np

from hsa_ros2 import rollout_ledger as rl


def test_trial_clock_starts_on_the_first_chunk_after_a_gap():
    """A chunk after idle_gap_s of silence starts a new trial; chunks inside the gap continue it."""
    c = rl.TrialClock(idle_gap_s=3.0)
    assert c.state(0.0) == 'waiting'
    assert c.on_chunk(10.0) is True
    assert c.on_chunk(10.8) is False
    assert c.chunks == 2 and c.state(11.0) == 'running'
    assert c.state(15.0) == 'stalled'
    assert c.on_chunk(20.0) is True and c.start == 20.0 and c.chunks == 1
    assert abs(c.mark(28.5) - 8.5) < 1e-9
    assert c.state(28.5) == 'waiting' and c.mark(30.0) == 0.0


def test_rows_round_trip_and_undo(tmp_path):
    """append -> read -> drop keeps COLUMNS and numbers rollouts past the largest one."""
    path = tmp_path / 'ledger.csv'
    assert rl.read_rows(path) == [] and rl.next_rollout([]) == 1
    p = rl.Placement.parse('P02 20 0 0')
    rl.append_row(path, rl.make_row(1, p, 'success', 8.44, checkpoint='ep025', bag='r_01'))
    rl.append_row(path, rl.make_row(5, p, 'failure', 30.0, failure_code='jam'))
    rows = rl.read_rows(path)
    assert list(rows[0].keys()) == list(rl.COLUMNS)
    assert rows[0]['time_s'] == '8.4' and rows[0]['dx_mm'] == '20' and rows[0]['bag'] == 'r_01'
    assert rl.next_rollout(rows) == 6
    assert rl.drop_last_row(path)['failure_code'] == 'jam'
    assert rl.next_rollout(rl.read_rows(path)) == 2
    rl.drop_last_row(path)
    assert rl.drop_last_row(path) is None


def test_placement_parse_keeps_unspecified_numbers():
    """``p`` with only an id keeps the offsets; a full line replaces them."""
    cur = rl.Placement('P01', 20.0, -5.0, 15.0)
    assert rl.Placement.parse('P03', cur) == rl.Placement('P03', 20.0, -5.0, 15.0)
    assert rl.Placement.parse('P04 0 10', cur) == rl.Placement('P04', 0.0, 10.0, 15.0)
    assert cur.label() == 'P01 (+20, -5, +15 deg)'
    try:
        rl.Placement.parse('')
    except ValueError:
        pass
    else:
        raise AssertionError('empty placement accepted')


def test_newest_bag(tmp_path):
    """The most recently modified directory, or '' with none."""
    assert rl.newest_bag(tmp_path / 'missing') == ''
    (tmp_path / 'r_01').mkdir()
    (tmp_path / 'r_02').mkdir()
    import os

    os.utime(tmp_path / 'r_01', (1, 1))
    assert rl.newest_bag(tmp_path) == 'r_02'


def test_status_line():
    """Waiting, running with force and plate, and the last row."""
    live = rl.Live()
    c = rl.TrialClock()
    assert rl.status_line(3, c, live, 0.0).startswith(
        '#3 P00 (+0, +0, +0 deg)  waiting for the first chunk  F  --  plate  --'
    )
    c.on_chunk(100.0)
    live.force = np.array([3.0, 4.0, 0.0, 0, 0, 0])
    live.plate_z_m = -0.0121
    live.last_row = rl.make_row(2, rl.Placement(), 'failure', 30.0, failure_code='miss')
    line = rl.status_line(3, c, live, 102.0)
    assert 'RUNNING   2.0 s  1 chunks' in line and 'F  5.0 N' in line and 'plate -12.1 mm' in line
    live.rest = 'REST OFF x-6.3'
    assert 'plate -12.1 mm  REST OFF x-6.3' in rl.status_line(3, c, live, 102.0)
    assert line.endswith('| last #2 failure miss 30.0 s  [s f p u q]')


def test_rest_check_names_the_axes_that_are_off():
    """Inside every tolerance is OK; each axis outside it is named with its offset."""
    check = rl.RestCheck((0.3, 2.9, -11.3), (2.6, 1.8, 1.8), 9.4, 3.4)
    assert check.evaluate(np.array([-2.0, 2.9, -12.1]), 8.0) == (True, 'rest OK')
    assert check.evaluate(np.array([-6.0, 2.9, -11.3]), 15.0) == (False, 'REST OFF x-6.3 rot+5.6')


def test_rest_pose_is_the_median_before_the_start():
    """Samples at or after the start and older than the window are ignored."""
    samples = [(t, np.array([t, 0.0, 0.0]), 10.0 + t) for t in (0.0, 9.2, 9.5, 9.8, 10.0)]
    pos, rot = rl.rest_pose(samples, until=10.0, window_s=1.0)
    assert np.allclose(pos, [9.5, 0, 0]) and rot == 19.5
    assert rl.rest_pose(samples, until=20.0, window_s=1.0) is None


def test_rotation_deg():
    """Identity is 0; a 90 deg turn about z is 90 either sign of w."""
    assert rl.rotation_deg(0, 0, 0, 1) == 0.0
    s = np.sqrt(0.5)
    assert np.isclose(rl.rotation_deg(0, 0, s, s), 90.0) and np.isclose(rl.rotation_deg(0, 0, -s, -s), 90.0)


def test_rest_columns_land_in_the_row(tmp_path):
    """The rest pose goes into the plate_rest_* columns; a row without a plate leaves them empty."""
    cols = rl.rest_columns((np.array([-6.04, 2.9, -11.3]), 9.0), rl.RestCheck())
    path = tmp_path / 'ledger.csv'
    rl.append_row(path, rl.make_row(1, rl.Placement(), 'failure', 20.0, rest=cols))
    rl.append_row(path, rl.make_row(2, rl.Placement(), 'success', 20.0))
    rows = rl.read_rows(path)
    assert rows[0]['plate_rest_x_mm'] == '-6.0' and rows[0]['plate_rest_ok'] == '0'
    assert rows[1]['plate_rest_ok'] == ''
