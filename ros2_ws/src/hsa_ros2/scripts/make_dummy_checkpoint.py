#!/usr/bin/env python
"""
Write a checkpoint in the diffusion-policy fork's own format WITHOUT training.

It brings up and times the serving path before a real checkpoint exists; its actions are meaningless.
Runs INSIDE the polyumi-dp image (the fork's workspace class and exact dependency tree are what make
the file loadable by serve_policy.py). From the HSA-UMI repo root:

    mkdir -p data/dp_outputs
    docker run --rm --shm-size=8g --user $(id -u):$(id -g) \
        -e HOME=/tmp -e HF_HOME=/hf_cache -e MPLCONFIGDIR=/tmp/mpl -e NUMBA_CACHE_DIR=/tmp/numba \
        -v /tmp/hsa_desk/hsa_desk.zarr.zip:/data/dataset.zarr.zip:ro \
        -v "$PWD/data/dp_outputs:/app/data/outputs:rw" -v "$HOME/.cache/huggingface:/hf_cache:rw" \
        -v "$PWD/ros2_ws/src/hsa_ros2/scripts/make_dummy_checkpoint.py:/app/make_dummy_checkpoint.py:ro" \
        polyumi-dp python make_dummy_checkpoint.py --out /app/data/outputs/dummy_random.ckpt --check

Plain ``python``, not ``micromamba run -n umi python``: the image's entrypoint activates the env, and
``micromamba run`` under ``--user $(id -u)`` fails on a lockfile it cannot write in mambauser's home.
The normalizer is fitted on a recorded .zarr.zip because predict_action cannot run without one; the
weights are the encoder's random init unless --pretrained is given (downloads the 600 MB CLIP ViT-B).
"""

import argparse
import pathlib
import sys
import time

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> int:
    """Build and verify the dummy checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='checkpoint path inside the container (parent must exist)')
    ap.add_argument('--dataset', default='/data/dataset.zarr.zip', help='recorder .zarr.zip to fit the normalizer on')
    ap.add_argument('--config-name', default='train_diffusion_unet_timm_polyumi_workspace')
    ap.add_argument('--pretrained', action='store_true', help='keep the CLIP ViT-B pretrained init (downloads ~600 MB)')
    ap.add_argument('--check', action='store_true', help='reload the file with serve_policy._load_policy afterwards')
    args = ap.parse_args()

    OmegaConf.register_new_resolver('eval', eval, replace=True)  # as train.py does
    overrides = [f'task.dataset_path={args.dataset}']
    if not args.pretrained:
        overrides.append('policy.obs_encoder.pretrained=false')
    with initialize_config_dir(config_dir='/app/diffusion_policy/config', version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    t0 = time.time()
    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg)
    print(f'workspace built in {time.time() - t0:.0f} s: {cfg._target_}')

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f'dataset: {len(dataset)} samples from {dataset.replay_buffer.n_episodes} episodes; fitting the normalizer')
    normalizer = dataset.get_normalizer()
    ws.model.set_normalizer(normalizer)
    if ws.ema_model is not None:
        ws.ema_model.set_normalizer(normalizer)

    out = pathlib.Path(args.out)
    ws.save_checkpoint(path=out, use_thread=False)
    print(f'wrote {out} ({out.stat().st_size / 2**20:.0f} MiB)')

    if args.check:
        sys.path.insert(0, '/app')
        from serve_policy import _load_policy

        policy, device, gripper_repr = _load_policy(str(out))
        n = sum(p.numel() for p in policy.parameters())
        print(
            f'reloaded through serve_policy._load_policy: {n / 1e6:.1f} M params on {device}, '
            f'gripper repr {gripper_repr}'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
