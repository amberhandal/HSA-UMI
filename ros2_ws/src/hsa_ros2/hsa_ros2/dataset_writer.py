"""
Writer for the UMI ReplayBuffer layout, the zarr v2 store the diffusion-policy fork trains on.

``data/<key>`` arrays are stacked along time, ``meta/episode_ends`` holds cumulative step counts,
and ``meta`` attrs carry provenance (inert to ``UmiDataset``). The key NAMES are load-bearing: the
sampler counts robots via ``key.endswith('eef_pos')`` and name-matches everything else, so they are
fixed here as constants. The frame behind ``robot0_eef_*`` is ``hsa_tcp``, the HSA wrist plate at the
neutral servo setpoint, published by the arm bringup as a fixed child of the flange (``fer_link8``)
and the frame the streaming controller commands; every consumer takes the frame name as a parameter
(``tcp_frame`` / ``eef_frame``), so the keys stay UMI's while the frame is ours.
"""

from __future__ import annotations

import json
import pathlib
import time
import zipfile

import numpy as np
import zarr
from numcodecs import Blosc

#: zstd level 5 with byte shuffle.
COMPRESSOR = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
#: The policy's image grid (camera0_rgb is [3, 224, 224] in shape_meta).
IMAGE_SIZE = 224
#: Time-chunk length for low-dimensional arrays. Images are chunked one frame per chunk.
LOWDIM_CHUNK = 1024

# The fixed key names; anything the recorder adds must go through here so a typo cannot silently
# produce a key the dataset never reads. Shapes and dtypes are in REQUIRED_KEYS / OPTIONAL_KEYS.
KEY_IMAGE = 'camera0_rgb'  # 224x224 RGB, the policy's grid
KEY_POS = 'robot0_eef_pos'  # TCP position in the base frame (fer_link0)
KEY_ROT = 'robot0_eef_rot_axis_angle'  # TCP rotation as a rotation vector
KEY_GRIPPER = 'robot0_gripper_width'  # 0.0 until a gripper exists (the sampler reads it)
KEY_START = 'robot0_demo_start_pose'  # the episode's first [pos, rotvec], repeated every step
KEY_END = 'robot0_demo_end_pose'  # the episode's last [pos, rotvec], repeated every step
KEY_ACTION = 'action'  # commanded teleop target [pos, rotvec, gripper]
KEY_WRIST_POS = 'robot0_wrist_servo'  # servo positions (rad), when the wrist is on
KEY_WRIST_CUR = 'robot0_wrist_current'  # servo currents (A), when the wrist is on
KEY_WRENCH = 'robot0_ext_wrench'  # Franka external wrench, when enabled
KEY_STAGE = 'stage'  # space-bar stage counter
KEY_TIME = 'timestamp'  # capture instant, wall-clock seconds

#: Keys every episode must carry, with their per-step shape and dtype.
REQUIRED_KEYS = {
    KEY_IMAGE: ((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8),
    KEY_POS: ((3,), np.float32),
    KEY_ROT: ((3,), np.float32),
    KEY_GRIPPER: ((1,), np.float32),
    KEY_START: ((6,), np.float32),
    KEY_END: ((6,), np.float32),
    KEY_ACTION: ((7,), np.float32),
    KEY_STAGE: ((1,), np.int64),
    KEY_TIME: ((), np.float64),
}
#: Keys that are present only when the corresponding stream is enabled.
OPTIONAL_KEYS = {
    KEY_WRIST_POS: ((6,), np.float32),
    KEY_WRIST_CUR: ((6,), np.float32),
    KEY_WRENCH: ((6,), np.float32),
}


# ----------------------------------------------------------------------------------------------
# Time alignment helpers (used by the recorder to sample every stream at the camera's instant)
# ----------------------------------------------------------------------------------------------


def interp_at(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray | None:
    """
    Linearly interpolate a stamped series to instant ``t``; hold the nearest end outside its span.

    :param times: (N,) ascending stamps in seconds.
    :param values: (N, D) samples.
    :returns: (D,) sample, or None if the series is empty.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return None
    values = np.asarray(values, dtype=np.float64)
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = int(np.searchsorted(times, t))
    t0, t1 = times[i - 1], times[i]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return values[i - 1] * (1.0 - a) + values[i] * a


def latest_before(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray | None:
    """Newest sample whose stamp is <= ``t`` (a zero-order hold), or None if none is that old."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return None
    i = int(np.searchsorted(times, t, side='right')) - 1
    return None if i < 0 else np.asarray(values)[i]


# ----------------------------------------------------------------------------------------------
# Episode assembly
# ----------------------------------------------------------------------------------------------


def assemble_episode(steps: list[dict]) -> dict[str, np.ndarray]:
    """
    Stack a list of per-step dicts into per-key arrays and add the start/end pose broadcasts.

    Each step dict carries the keys in :data:`REQUIRED_KEYS` except ``robot0_demo_start_pose`` /
    ``robot0_demo_end_pose`` (derived here), plus any :data:`OPTIONAL_KEYS` -- and the optional set
    must be the same on every step of the episode.
    """
    if not steps:
        raise ValueError('an episode needs at least one step')
    keys = set(steps[0])
    for i, s in enumerate(steps):
        if set(s) != keys:
            raise ValueError(f'step {i} carries keys {sorted(set(s))}, step 0 carried {sorted(keys)}')
    out: dict[str, np.ndarray] = {}
    specs = {**REQUIRED_KEYS, **OPTIONAL_KEYS}
    for key in keys:
        shape, dtype = specs[key]
        arr = np.stack([np.asarray(s[key], dtype=dtype).reshape(shape) for s in steps]).astype(dtype)
        out[key] = arr
    pose0 = np.concatenate([out[KEY_POS][0], out[KEY_ROT][0]]).astype(np.float32)
    pose1 = np.concatenate([out[KEY_POS][-1], out[KEY_ROT][-1]]).astype(np.float32)
    n = len(steps)
    out[KEY_START] = np.repeat(pose0[None], n, axis=0)
    out[KEY_END] = np.repeat(pose1[None], n, axis=0)
    missing = set(REQUIRED_KEYS) - set(out)
    if missing:
        raise ValueError(f'episode is missing required keys {sorted(missing)}')
    return out


# ----------------------------------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------------------------------


class ReplayBufferWriter:
    """
    Append-only UMI ReplayBuffer on disk, one directory store per dataset.

    Opened in append mode, so a recorder restart continues the same dataset; every episode is
    committed to disk as it ends, so a crash costs at most the episode in progress.
    """

    def __init__(self, output_dir: str | pathlib.Path, attrs: dict | None = None):
        """Create the writer for the dataset directory ``output_dir``."""
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zarr_path = self.output_dir / 'replay_buffer.zarr'
        self.root = zarr.open_group(str(self.zarr_path), mode='a')
        self.data = self.root.require_group('data')
        self.meta = self.root.require_group('meta')
        if 'episode_ends' not in self.meta:
            self.meta.create_dataset('episode_ends', shape=(0,), chunks=(1024,), dtype=np.int64, compressor=COMPRESSOR)
        if attrs:
            self.meta.attrs.update(attrs)

    # ---- read side ------------------------------------------------------------------------
    @property
    def episode_ends(self) -> np.ndarray:
        """Return the cumulative step index at which each episode ends."""
        return np.asarray(self.meta['episode_ends'][:], dtype=np.int64)

    @property
    def n_episodes(self) -> int:
        """Return the number of episodes written."""
        return int(self.meta['episode_ends'].shape[0])

    @property
    def n_steps(self) -> int:
        """Return the number of steps written."""
        ends = self.episode_ends
        return int(ends[-1]) if ends.size else 0

    def keys(self) -> list[str]:
        """Return the data keys present."""
        return sorted(self.data.array_keys())

    def summary(self) -> str:
        """Return a one-line summary of the dataset."""
        ends = self.episode_ends
        lengths = np.diff(np.concatenate([[0], ends])) if ends.size else np.array([], dtype=np.int64)
        lines = [f'{self.zarr_path}: {self.n_episodes} episode(s), {self.n_steps} steps']
        if lengths.size:
            lines.append(
                f'  episode lengths: min {lengths.min()} / median {int(np.median(lengths))} / max {lengths.max()}'
            )
        for key in self.keys():
            a = self.data[key]
            lines.append(f'  data/{key:28s} {str(a.shape):20s} {a.dtype}')
        return '\n'.join(lines)

    # ---- write side -----------------------------------------------------------------------
    def _ensure_array(self, key: str, example: np.ndarray) -> zarr.Array:
        """Create ``data/<key>`` on first use, with chunks along time and the shared compressor."""
        if key in self.data:
            arr = self.data[key]
            if arr.shape[1:] != example.shape[1:] or arr.dtype != example.dtype:
                raise ValueError(
                    f'data/{key} exists as {arr.shape[1:]} {arr.dtype} but this episode brings '
                    f'{example.shape[1:]} {example.dtype}; a dataset cannot mix schemas'
                )
            return arr
        per_step = example.shape[1:]
        chunks = (1, *per_step) if example.ndim == 4 else (LOWDIM_CHUNK, *per_step)
        return self.data.create_dataset(
            key, shape=(0, *per_step), chunks=chunks, dtype=example.dtype, compressor=COMPRESSOR
        )

    def add_episode(self, arrays: dict[str, np.ndarray]) -> int:
        """
        Append one episode (all arrays share the leading length) and return its index.

        Refuses a key set that differs from the store's, so every episode of a dataset carries
        the same columns -- the sampler indexes them positionally.
        """
        lengths = {k: int(v.shape[0]) for k, v in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f'all arrays must share a length, got {lengths}')
        n = next(iter(lengths.values()))
        if n == 0:
            raise ValueError('refusing to add an empty episode')
        existing = set(self.keys())
        if existing and existing != set(arrays):
            raise ValueError(f'store has keys {sorted(existing)}, episode has {sorted(arrays)}')
        for key, value in arrays.items():
            self._ensure_array(key, value).append(value)
        self.meta['episode_ends'].append(np.array([self.n_steps + n], dtype=np.int64))
        return self.n_episodes - 1

    def drop_last_episode(self) -> int | None:
        """Remove the most recent episode by resizing every array; returns its former index."""
        ends = self.episode_ends
        if ends.size == 0:
            return None
        start = int(ends[-2]) if ends.size >= 2 else 0
        for key in self.keys():
            arr = self.data[key]
            arr.resize((start, *arr.shape[1:]))
        self.meta['episode_ends'].resize((ends.size - 1,))
        return int(ends.size - 1)

    def write_sidecar(self, episode_idx: int, record: dict) -> pathlib.Path:
        """Write one JSON file of provenance per episode (grid cell, wrist set, drops, ...)."""
        d = self.output_dir / 'episodes'
        d.mkdir(exist_ok=True)
        path = d / f'episode_{episode_idx:04d}.json'
        path.write_text(json.dumps(record, indent=2, default=str))
        return path

    def export_zip(self, out_path: str | pathlib.Path | None = None) -> pathlib.Path:
        """
        Pack the directory store into a ``.zarr.zip`` readable by ``zarr.ZipStore`` / ``UmiDataset``.

        The zip holds the store's *contents* at its root (``.zgroup``, ``data/...``); a top-level folder inside
        the zip would break ZipStore. Chunks are already Blosc-compressed, so the zip itself stores them uncompressed.
        """
        out = pathlib.Path(out_path) if out_path else self.output_dir / f'{self.output_dir.name}.zarr.zip'
        tmp = out.with_suffix(out.suffix + '.tmp')
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for f in sorted(self.zarr_path.rglob('*')):
                if f.is_file():
                    zf.write(f, f.relative_to(self.zarr_path).as_posix())
        tmp.replace(out)
        return out


def open_replay_buffer(path: str | pathlib.Path) -> zarr.Group:
    """
    Open a dataset read-only.

    ``path`` is a session directory (its ``replay_buffer.zarr``), the zarr directory itself, or an
    exported ``.zarr.zip``.
    """
    p = pathlib.Path(path).expanduser()
    if p.is_dir() and (p / 'replay_buffer.zarr').exists():
        p = p / 'replay_buffer.zarr'
    if p.suffix == '.zip':
        return zarr.group(zarr.ZipStore(str(p), mode='r'))
    return zarr.open_group(str(p), mode='r')


def open_zip_summary(zip_path: str | pathlib.Path) -> str:
    """Reopen an exported ``.zarr.zip`` the way UmiDataset does and describe it (a sanity check)."""
    with zarr.ZipStore(str(zip_path), mode='r') as store:
        root = zarr.group(store)
        ends = np.asarray(root['meta/episode_ends'][:])
        lines = [f'{zip_path}: {ends.size} episode(s), {int(ends[-1]) if ends.size else 0} steps']
        for key in sorted(root['data'].array_keys()):
            a = root['data'][key]
            lines.append(f'  data/{key:28s} {str(a.shape):20s} {a.dtype}')
        lines.append(f'  meta attrs: {dict(root["meta"].attrs)}')
    return '\n'.join(lines)


def now_iso() -> str:
    """Return the current time as an ISO 8601 string."""
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')
