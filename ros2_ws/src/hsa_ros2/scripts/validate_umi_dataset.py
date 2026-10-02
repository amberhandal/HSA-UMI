#!/usr/bin/env python
"""
Load a recorded .zarr.zip with UMI's real ``UmiDataset`` and check it yields the shapes training expects.

The ROS-free proof that what demo_recorder wrote is trainable. Needs torch, zarr 2.x, imagecodecs and
the UMI fork's bundled ``diffusion_policy`` package, i.e. the analysis venv:

    ~/hsa-ws/diffusion_policy/venv/bin/python scripts/validate_umi_dataset.py /path/to/name.zarr.zip
    ... --umi-fork ~/hsa-ws/universal_manipulation_interface_hsa   (default)
    ... --down-sample 1                                             (if recorded with store_every_n_frames: 3)

Expected tail of the output:  action: (16, 10) ... VALIDATION OK
"""

import argparse
import pathlib
import sys

import numpy as np
import zarr


def main() -> int:
    """Validate the dataset named on the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('zip_path')
    ap.add_argument(
        '--umi-fork',
        default='~/hsa-ws/universal_manipulation_interface_hsa',
        help='checkout whose bundled diffusion_policy/ holds UmiDataset',
    )
    ap.add_argument(
        '--down-sample', type=int, default=3, help='obs/action down_sample_steps: 3 for 30 Hz storage, 1 for 10 Hz'
    )
    args = ap.parse_args()

    # The fork's bundled package must win over any other diffusion_policy on the path.
    sys.path.insert(0, str(pathlib.Path(args.umi_fork).expanduser()))

    with zarr.ZipStore(args.zip_path, mode='r') as store:
        root = zarr.group(store)
        ends = root['meta/episode_ends'][:]
        pos = root['data/robot0_eef_pos'][:]
        ts = root['data/timestamp'][:]
        print(
            f'episodes {ends.tolist()} | eef_pos spread {np.ptp(pos, axis=0).round(4).tolist()} | '
            f'median dt {np.median(np.diff(ts[: ends[0]])):.4f} s | keys {sorted(root["data"].array_keys())}'
        )

    from diffusion_policy.dataset.umi_dataset import UmiDataset

    ds_steps = args.down_sample

    def lowdim(raw, shape=None, rot=False):
        d = dict(shape=shape or raw, horizon=2, latency_steps=0, down_sample_steps=ds_steps, type='low_dim')
        if rot:
            d['raw_shape'] = raw
            d['rotation_rep'] = 'rotation_6d'
        return d

    shape_meta = {
        'obs': {
            'camera0_rgb': dict(
                shape=[3, 224, 224],
                horizon=2,
                latency_steps=0,
                down_sample_steps=ds_steps,
                type='rgb',
                ignore_by_policy=False,
            ),
            'robot0_eef_pos': lowdim([3]),
            'robot0_eef_rot_axis_angle': lowdim([3], [6], rot=True),
            'robot0_gripper_width': lowdim([1]),
            'robot0_eef_rot_axis_angle_wrt_start': lowdim([3], [6], rot=True),
        },
        'action': dict(shape=[10], horizon=16, latency_steps=0, down_sample_steps=ds_steps, rotation_rep='rotation_6d'),
    }
    ds = UmiDataset(
        shape_meta=shape_meta,
        dataset_path=args.zip_path,
        cache_dir=None,
        pose_repr={'obs_pose_repr': 'relative', 'action_pose_repr': 'relative'},
        val_ratio=0.0,
    )
    print(f'UmiDataset: {len(ds)} training samples from {ds.replay_buffer.n_episodes} episodes')
    if len(ds) == 0:
        print('no samples: every episode is shorter than (16-1)*down_sample+1 steps')
        return 1
    s = ds[0]
    print('obs keys:', {k: tuple(v.shape) for k, v in s['obs'].items()})
    print('action:', tuple(s['action'].shape), 'dtype', s['action'].dtype)
    assert tuple(s['action'].shape) == (16, 10), 'action must be [16, 10]'
    assert tuple(s['obs']['camera0_rgb'].shape) == (2, 3, 224, 224)
    print('VALIDATION OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
