"""EpisodeVideo: frames go to a temporary file, then become episode_NNNN.mp4 + frame stamps, or vanish."""

import csv

import cv2
import numpy as np

from hsa_ros2.episode_video import EpisodeVideo, delete_episode_video


def _frame(i):
    f = np.zeros((48, 64, 3), dtype=np.uint8)
    f[:, :, 0] = (i * 20) % 256
    return f


def test_finish_writes_a_playable_mp4_and_frame_stamps(tmp_path):
    v = EpisodeVideo(tmp_path / 'videos', fps=30.0)
    for i in range(12):
        v.write(_frame(i), 100.0 + i / 30.0)
    out = v.finish('episode_0003')
    assert out == tmp_path / 'videos' / 'episode_0003.mp4' and out.exists()
    assert not (tmp_path / 'videos' / '.recording.mp4').exists()
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (64, 48)
    rows = list(csv.DictReader(open(tmp_path / 'videos' / 'episode_0003_frames.csv')))
    assert len(rows) == 12 and float(rows[0]['stamp_s']) == 100.0


def test_discard_and_delete_leave_nothing(tmp_path):
    v = EpisodeVideo(tmp_path / 'videos')
    v.write(_frame(0), 1.0)
    v.discard()
    assert not any((tmp_path / 'videos').iterdir())
    assert v.finish('episode_0000') is None, 'nothing written since the discard'
    v.write(_frame(1), 2.0)
    v.finish('episode_0001')
    delete_episode_video(tmp_path / 'videos', 'episode_0001')
    assert not any((tmp_path / 'videos').iterdir())
