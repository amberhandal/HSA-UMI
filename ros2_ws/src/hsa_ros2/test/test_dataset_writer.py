"""
Round-trip tests for the ROS-free ReplayBuffer writer (needs numpy + zarr 2.x).

    cd ~/hsa-ws/HSA-UMI/ros2_ws/src/hsa_ros2 && PYTHONPATH=. pytest test/test_dataset_writer.py -q
"""

import numpy as np
import pytest
import zarr

from hsa_ros2 import dataset_writer as dw


def _steps(n, t0=0.0, x0=0.5, with_optional=False):
    steps = []
    for i in range(n):
        s = {
            dw.KEY_IMAGE: np.full((dw.IMAGE_SIZE, dw.IMAGE_SIZE, 3), i % 256, dtype=np.uint8),
            dw.KEY_POS: np.array([x0 + 0.001 * i, 0.0, 0.3]),
            dw.KEY_ROT: np.array([np.pi, 0.0, 0.0]),
            dw.KEY_GRIPPER: np.zeros(1),
            dw.KEY_ACTION: np.array([x0 + 0.001 * (i + 1), 0.0, 0.3, np.pi, 0.0, 0.0, 0.0]),
            dw.KEY_STAGE: np.array([i // 10]),
            dw.KEY_TIME: np.array(t0 + i / 30.0),
        }
        if with_optional:
            s[dw.KEY_WRIST_EXT] = np.full(6, 360.0)
            s[dw.KEY_WRIST_IQ] = np.full(6, 2.0)
            s[dw.KEY_WRIST_ERRORS] = np.zeros(6, dtype=np.int64)
            s[dw.KEY_PLATE_POS] = np.array([0.0, 0.0, 0.012])
            s[dw.KEY_PLATE_ROT] = np.zeros(3)
            s[dw.KEY_PLATE_VISIBLE] = np.ones(1, dtype=np.int64)
            s[dw.KEY_WRIST_IMAGE] = np.full((dw.IMAGE_SIZE, dw.IMAGE_SIZE, 3), 7, dtype=np.uint8)
        steps.append(s)
    return steps


def test_assemble_adds_start_end_broadcasts():
    """Check that assemble adds start and end and broadcasts scalars."""
    arrays = dw.assemble_episode(_steps(5))
    assert arrays[dw.KEY_START].shape == (5, 6) and arrays[dw.KEY_END].shape == (5, 6)
    assert np.allclose(arrays[dw.KEY_START][3], np.concatenate([arrays[dw.KEY_POS][0], arrays[dw.KEY_ROT][0]]))
    assert np.allclose(arrays[dw.KEY_END][0], np.concatenate([arrays[dw.KEY_POS][-1], arrays[dw.KEY_ROT][-1]]))
    assert arrays[dw.KEY_IMAGE].dtype == np.uint8 and arrays[dw.KEY_ACTION].dtype == np.float32


def test_writer_round_trip_and_drop(tmp_path):
    """Check a write, read back and drop round trip."""
    w = dw.ReplayBufferWriter(tmp_path / 'ds', attrs={'source': 'test'})
    assert w.add_episode(dw.assemble_episode(_steps(50))) == 0
    assert w.add_episode(dw.assemble_episode(_steps(60, t0=10.0))) == 1
    assert w.n_episodes == 2 and w.n_steps == 110
    assert list(w.episode_ends) == [50, 110]
    # reopen in append mode: the same dataset continues
    w2 = dw.ReplayBufferWriter(tmp_path / 'ds')
    assert w2.n_episodes == 2
    assert w2.drop_last_episode() == 1
    assert w2.n_episodes == 1 and w2.n_steps == 50 and w2.data[dw.KEY_IMAGE].shape[0] == 50
    out = w2.export_zip()
    with zarr.ZipStore(str(out), mode='r') as store:
        root = zarr.group(store)
        assert list(root['meta/episode_ends'][:]) == [50]
        assert root['data'][dw.KEY_IMAGE].shape == (50, 224, 224, 3)
        assert root['data'][dw.KEY_ACTION].shape == (50, 7)
        assert root['meta'].attrs['source'] == 'test'
    assert 'episode' in dw.open_zip_summary(out)


def test_schema_mismatch_is_refused(tmp_path):
    """Check that a schema mismatch is refused."""
    w = dw.ReplayBufferWriter(tmp_path / 'ds')
    w.add_episode(dw.assemble_episode(_steps(3)))
    try:
        w.add_episode(dw.assemble_episode(_steps(3, with_optional=True)))
    except ValueError as e:
        assert 'keys' in str(e)
    else:
        raise AssertionError('mixing key sets must be refused')


def test_interp_and_hold():
    """Check interpolation and hold."""
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    assert np.allclose(dw.interp_at(times, values, 0.5), [0.5, 15.0])
    assert np.allclose(dw.interp_at(times, values, -1.0), [0.0, 10.0])
    assert np.allclose(dw.interp_at(times, values, 5.0), [2.0, 30.0])
    assert np.allclose(dw.latest_before(times, values, 1.5), [1.0, 20.0])
    assert dw.latest_before(times, values, -0.1) is None
    assert dw.interp_at(np.array([]), np.zeros((0, 2)), 0.0) is None


def test_failures_are_left_out_of_the_export(tmp_path):
    """Episodes saved as failures stay in the store but never reach the training zip."""
    w = dw.ReplayBufferWriter(tmp_path / 'ds', attrs={'source': 'test'})
    for i, (n, outcome) in enumerate([(10, 'success'), (7, 'failure'), (12, 'success')]):
        idx = w.add_episode(dw.assemble_episode(_steps(n, t0=10.0 * i, x0=0.1 * i)))
        w.write_sidecar(idx, {'episode': idx, 'outcome': outcome})
    assert w.episode_outcomes() == ['success', 'failure', 'success']

    out = w.export_zip(tmp_path / 'train.zarr.zip', episodes=[0, 2])
    with zarr.ZipStore(str(out), mode='r') as store:
        root = zarr.group(store)
        assert root['meta/episode_ends'][:].tolist() == [10, 22]
        pos = root['data/robot0_eef_pos'][:]
        assert np.allclose(pos[:10], w.data['robot0_eef_pos'][:10])
        assert np.allclose(pos[10:], w.data['robot0_eef_pos'][17:29])
        assert root['meta'].attrs['source_episodes'] == [0, 2]
    assert w.n_episodes == 3, 'the store itself keeps the failure'
    assert not list((tmp_path / 'ds').glob('.export_*')), 'the temporary subset store is cleaned up'


def test_missing_outcome_counts_as_success(tmp_path):
    """Datasets recorded before outcomes existed export unchanged."""
    w = dw.ReplayBufferWriter(tmp_path / 'ds')
    w.add_episode(dw.assemble_episode(_steps(5)))
    w.write_sidecar(0, {'episode': 0})
    w.add_episode(dw.assemble_episode(_steps(5, t0=5.0)))
    assert w.episode_outcomes() == ['success', 'success']



def test_stage_segments_and_names():
    """Contiguous runs of a stage value, and names or indices both resolve."""
    assert dw.stage_segments(np.array([0, 0, 1, 1, 1, 2, 0])) == [(0, 0, 2), (1, 2, 5), (2, 5, 6), (0, 6, 7)]
    assert dw.stage_index('turn', dw.DEFAULT_STAGE_NAMES) == 1
    assert dw.stage_index('2', dw.DEFAULT_STAGE_NAMES) == 2
    with pytest.raises(ValueError):
        dw.stage_index('wiggle', dw.DEFAULT_STAGE_NAMES)


def test_export_one_subtask_recomputes_the_start_pose(tmp_path):
    """--stage turn: each turn segment is its own episode, and its start pose is the segment's first pose."""
    w = dw.ReplayBufferWriter(tmp_path / 'ds')
    for i in range(2):
        steps = _steps(30, t0=10.0 * i, x0=0.1 * i)
        for k, st in enumerate(steps):
            st[dw.KEY_STAGE] = np.array([0 if k < 10 else 1 if k < 22 else 2])
        w.add_episode(dw.assemble_episode(steps))
    out = w.export_zip(tmp_path / 'turn.zarr.zip', stage=1)
    with zarr.ZipStore(str(out), mode='r') as store:
        root = zarr.group(store)
        assert root['meta/episode_ends'][:].tolist() == [12, 24]
        assert set(root['data/stage'][:].reshape(-1).tolist()) == {1}
        start = root['data/robot0_demo_start_pose'][:]
        pos = root['data/robot0_eef_pos'][:]
        assert np.allclose(start[0, :3], pos[0]) and np.allclose(start[12, :3], pos[12])
        assert root['meta'].attrs['source_segments'] == [[0, 10, 22], [1, 10, 22]]
    with pytest.raises(ValueError):
        w.export_zip(tmp_path / 'none.zarr.zip', stage=5)



def test_ledger_csv_has_one_row_per_episode_with_subtask_times(tmp_path):
    """export_dataset --csv: outcome, stopped_by and per-subtask seconds per episode."""
    import csv

    from hsa_ros2 import export_dataset

    w = dw.ReplayBufferWriter(tmp_path / 'ds')
    for i, (outcome, by) in enumerate([('success', 'operator'), ('failure', 'timeout')]):
        idx = w.add_episode(dw.assemble_episode(_steps(10, t0=10.0 * i)))
        w.write_sidecar(idx, {'episode': idx, 'outcome': outcome, 'stopped_by': by, 'n_steps': 10, 'duration_s': 0.3,
                              'stage_names': ['insert', 'turn', 'open'],
                              'stages': [{'stage': 0, 'name': 'insert', 'start_step': 0, 'end_step': 10,
                                          'duration_s': 0.3}]})  # fmt: skip
    out = tmp_path / 'ledger.csv'
    assert export_dataset.main([str(tmp_path / 'ds'), '--csv', str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert [r['outcome'] for r in rows] == ['success', 'failure']
    assert rows[1]['stopped_by'] == 'timeout' and rows[0]['insert_s'] == '0.3' and rows[0]['turn_s'] == '0.0'


class _FakeWristState:
    extension_deg = [360.0] * 6
    command_deg = [360.0] * 6
    motor_pos_turns = [1.0, 2.0]  # short on purpose
    motor_vel_turns_s = [0.0] * 6
    torque_estimate_nm = [0.1] * 6
    torque_target_nm = [0.0] * 6
    iq_setpoint_a = [0.0] * 6
    iq_measured_a = [2.0] * 6
    bus_voltage_v = [24.0] * 6
    bus_current_a = [0.5] * 6
    fet_temperature_c = [30.0] * 6
    motor_temperature_c = [35.0] * 6
    controller_errors = [0, 0, 0, 4, 0, 0]
    axis_state = [8] * 7  # long on purpose


def test_wrist_state_arrays_cover_every_wrist_key_with_six_columns():
    """Every wrist key gets a (6,) column from its message field, padded or trimmed."""
    arrays = dw.wrist_state_arrays(_FakeWristState())
    assert set(arrays) == set(dw.WRIST_STATE_FIELDS)
    assert all(a.shape == (6,) for a in arrays.values())
    assert arrays[dw.KEY_WRIST_EXT][0] == 360.0 and arrays[dw.KEY_WRIST_IQ][5] == 2.0
    assert arrays[dw.KEY_WRIST_MOTOR_POS][1] == 2.0 and np.isnan(arrays[dw.KEY_WRIST_MOTOR_POS][2])
    assert arrays[dw.KEY_WRIST_ERRORS][3] == 4 and arrays[dw.KEY_WRIST_AXIS_STATE].tolist() == [8] * 6
    # and they assemble into an episode with the declared dtypes
    steps = _steps(3)
    for s in steps:
        s.update(arrays)
    out = dw.assemble_episode(steps)
    assert out[dw.KEY_WRIST_ERRORS].dtype == np.int64 and out[dw.KEY_WRIST_IQ].dtype == np.float32


def test_wrist_and_plate_keys_round_trip_through_the_store(tmp_path):
    """The wrist, plate and second-camera keys are written, re-read and exported with the rest."""
    pytest.importorskip('cv2')
    w = dw.ReplayBufferWriter(tmp_path / 'd')
    w.add_episode(dw.assemble_episode(_steps(4, with_optional=True)))
    assert dw.KEY_WRIST_IMAGE in w.keys() and w.data[dw.KEY_WRIST_IMAGE].shape == (4, 224, 224, 3)
    assert w.data[dw.KEY_PLATE_POS][:, 2].tolist() == pytest.approx([0.012] * 4)
    assert w.data[dw.KEY_PLATE_VISIBLE].dtype == np.int64
    assert w.data[dw.KEY_WRIST_EXT][0].tolist() == [360.0] * 6


def test_resize_wrist_rgb_keeps_the_whole_frame():
    """A 1280x800 frame becomes 224x224 without a crop: both edges survive."""
    cv2 = pytest.importorskip('cv2')
    img = np.zeros((800, 1280, 3), dtype=np.uint8)
    img[:, :40] = 255  # a bar at the left edge, which a 4:3 crop would remove
    img[:, -40:] = 255
    out = dw.resize_wrist_rgb(img)
    assert out.shape == (224, 224, 3) and out.flags['C_CONTIGUOUS']
    assert out[100, 0, 0] > 200 and out[100, -1, 0] > 200 and out[100, 112, 0] == 0
    del cv2
