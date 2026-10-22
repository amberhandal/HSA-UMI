"""
One video file per recorded episode: full-resolution camera frames plus the capture time of each.

ROS-free (OpenCV + numpy), so it is testable without a camera. The recorder opens a temporary file when an
episode starts, appends every camera frame while recording, and on save renames it to
``videos/episode_NNNN.mp4`` next to ``episode_NNNN_frames.csv`` (frame index, header stamp in seconds);
a discarded episode deletes the temporary file.
"""

from __future__ import annotations

import csv
import pathlib

import cv2
import numpy as np


class EpisodeVideo:
    """Write RGB frames to an mp4 under a temporary name until :meth:`finish` or :meth:`discard`."""

    def __init__(self, directory: str | pathlib.Path, fps: float = 30.0):
        """Prepare a writer in ``directory`` (created if missing); the file opens with the first frame."""
        self.dir = pathlib.Path(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.tmp = self.dir / '.recording.mp4'
        self._writer = None
        self._stamps: list[float] = []

    @property
    def n_frames(self) -> int:
        """Return the number of frames written so far."""
        return len(self._stamps)

    def write(self, rgb: np.ndarray, stamp_s: float) -> None:
        """Append one HxWx3 uint8 RGB frame captured at ``stamp_s``."""
        if self._writer is None:
            h, w = rgb.shape[:2]
            self._writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f'could not open a video writer at {self.tmp}')
        self._writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        self._stamps.append(float(stamp_s))

    def finish(self, name: str) -> pathlib.Path | None:
        """Close the file as ``<name>.mp4`` with ``<name>_frames.csv``; return the mp4 path (None if empty)."""
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        out = self.dir / f'{name}.mp4'
        self.tmp.replace(out)
        with open(self.dir / f'{name}_frames.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'stamp_s'])
            w.writerows(enumerate(self._stamps))
        self._stamps = []
        return out

    def discard(self) -> None:
        """Drop the episode in progress and its temporary file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.tmp.unlink(missing_ok=True)
        self._stamps = []


def delete_episode_video(directory: str | pathlib.Path, name: str) -> None:
    """Remove a saved episode's video and frame list (used when a saved episode is dropped)."""
    d = pathlib.Path(directory).expanduser()
    for f in (d / f'{name}.mp4', d / f'{name}_frames.csv'):
        f.unlink(missing_ok=True)
