"""
Round-trip tests for the ROS-free ReplayBuffer writer (needs numpy + zarr 2.x).

    cd ~/hsa-ws/HSA-UMI/ros2_ws/src/hsa_ros2 && PYTHONPATH=. pytest test/test_dataset_writer.py -q
"""

import numpy as np
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
            s[dw.KEY_WRIST_POS] = np.zeros(6)
            s[dw.KEY_WRIST_CUR] = np.zeros(6)
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
