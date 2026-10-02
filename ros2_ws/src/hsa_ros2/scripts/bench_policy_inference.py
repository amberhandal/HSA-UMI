#!/usr/bin/env python
"""
Measure what a diffusion policy costs at inference time on THIS machine's GPU.

Batch-1 ``predict_action`` timings (GPU synchronised), parameter counts and peak memory. ``old-ckpt``
times the polymetis-era checkpoint (ResNet18 + UNet 512/1024/2048 at 240x320); ``new-arch`` times the
UMI fork's ``diffusion_unet_timm_policy`` (CLIP ViT-B/16 + UNet 256/512/1024 at 224x224) with random
weights (pretrained=false, no download), which costs the same as a trained one, so it predicts a
lamb-trained checkpoint before one exists.

    PY=~/hsa-ws/diffusion_policy/venv/bin/python   # the analysis venv: torch + CUDA; timm installed with --no-deps
    $PY bench_policy_inference.py old-ckpt ~/hsa-ws/checkpoints/peg_insertion_v1/epoch=0350-train_loss=0.002.ckpt
    $PY bench_policy_inference.py new-arch [--umi-fork ~/hsa-ws/universal_manipulation_interface_hsa]

Measured 2026-09-29 on the dev laptop (RTX 500 Ada, 4 GB, torch 2.12 cu130):
    old-ckpt   288.6 M params   587 ms @ DDIM 16   296 ms @ 8   423 ms bf16   peak 1.2 GiB
    new-arch   171.3 M params   207 ms @ DDIM 16   120 ms @ 8   140 ms bf16   peak 0.7 GiB
"""

from __future__ import annotations

import argparse
import pathlib
import resource
import sys
import time

import numpy as np
import torch


def bench(policy, obs, steps, n=20, autocast=None):
    """Time predict_action n times after a warm-up; returns milliseconds per call."""
    policy.num_inference_steps = steps
    ctx = torch.autocast('cuda', dtype=autocast) if autocast else torch.no_grad()
    with torch.no_grad(), ctx:
        for _ in range(3):
            policy.predict_action(obs)
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            policy.predict_action(obs)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    return np.array(ts) * 1e3


def report(label, ms):
    """Print timing statistics for ``ms``."""
    print(
        f'  {label:36s} mean {ms.mean():6.1f} ms   p50 {np.median(ms):6.1f}   '
        f'p95 {np.percentile(ms, 95):6.1f}   max {ms.max():6.1f}'
    )


def mem(label):
    """Print the memory in use under ``label``."""
    free, total = torch.cuda.mem_get_info()
    print(
        f'  {label:36s} peak allocated {torch.cuda.max_memory_allocated() / 2**20:5.0f} MiB   '
        f'device in use {(total - free) / 2**20:5.0f} / {total / 2**20:.0f} MiB (includes the CUDA context)'
    )


def params(policy):
    """Print the parameter counts of ``policy``."""
    tot = sum(p.numel() for p in policy.parameters())
    enc = sum(p.numel() for p in policy.obs_encoder.parameters()) if hasattr(policy, 'obs_encoder') else 0
    unet = sum(p.numel() for p in policy.model.parameters()) if hasattr(policy, 'model') else 0
    print(
        f'  parameters: total {tot / 1e6:.1f} M (encoder {enc / 1e6:.1f} M, UNet {unet / 1e6:.1f} M); '
        f'fp32 weights {tot * 4 / 2**20:.0f} MiB'
    )


def make_obs(shape_meta, dev):
    """Build a dummy observation for ``shape_meta``."""
    obs = {}
    for k, v in shape_meta['obs'].items():
        shp = tuple(v['shape'])
        horizon = int(v.get('horizon', 2))
        obs[k] = (torch.rand if v.get('type', 'low_dim') == 'rgb' else torch.zeros)(1, horizon, *shp, device=dev)
    print('  obs:', {k: tuple(t.shape) for k, t in obs.items()})
    return obs


def run_suite(policy, obs):
    """Time the policy on ``obs``."""
    torch.cuda.reset_peak_memory_stats()
    report('fp32, DDIM 16 steps (default)', bench(policy, obs, 16))
    mem('after the 16-step runs')
    report('fp32, DDIM 8 steps', bench(policy, obs, 8))
    report('bf16 autocast, DDIM 16 steps', bench(policy, obs, 16, autocast=torch.bfloat16))


def old_ckpt(path: str):
    """Load the old-stack checkpoint at ``path``."""
    import dill
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver('eval', eval, replace=True)
    print(f'== old checkpoint: {pathlib.Path(path).name}')
    t0 = time.time()
    # weights_only=False is REQUIRED on torch >= 2.6: the checkpoint is a dill pickle of the whole
    # workspace payload, which the new default refuses. The old eval script does not pass it.
    payload = torch.load(open(path, 'rb'), pickle_module=dill, map_location='cpu', weights_only=False)
    cfg = payload['cfg']
    ws = hydra.utils.get_class(cfg._target_)(cfg)
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    del payload
    print(
        f'  loaded + built in {time.time() - t0:.0f} s; '
        f'CPU RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB'
    )
    dev = torch.device('cuda')
    policy.eval().to(dev)
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    params(policy)
    sm = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    for v in sm['obs'].values():
        v['horizon'] = cfg.n_obs_steps
    run_suite(policy, make_obs(sm, dev))


def new_arch(umi_root: str):
    """Build the new architecture from the fork at ``umi_root``."""
    sys.path.insert(0, umi_root)  # the fork's bundled diffusion_policy must win over the standalone one
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver('eval', eval, replace=True)
    print('== PolyUMI-fork architecture with random weights (UMI fork: diffusion_unet_timm_policy)')
    with initialize_config_dir(config_dir=f'{umi_root}/diffusion_policy/config', version_base=None):
        cfg = compose(
            config_name='train_diffusion_unet_timm_umi_workspace', overrides=['policy.obs_encoder.pretrained=false']
        )
    p = cfg.policy
    print(
        f'  encoder {p.obs_encoder.model_name} | UNet down_dims {list(p.down_dims)} | '
        f'DDIM train {p.noise_scheduler.num_train_timesteps} infer {p.num_inference_steps} | '
        f'action horizon {OmegaConf.select(cfg, "task.action_horizon")} '
        f'obs horizon {OmegaConf.select(cfg, "task.img_obs_horizon")}'
    )
    policy = hydra.utils.instantiate(p)
    from diffusion_policy.common.normalize_util import get_image_identity_normalizer
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

    sm = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    norm = (
        LinearNormalizer()
    )  # predict_action normalises first; a fitted-on-noise normalizer stands in for a trained one
    for k, v in sm['obs'].items():
        norm[k] = (
            get_image_identity_normalizer()
            if v['type'] == 'rgb'
            else SingleFieldLinearNormalizer.create_fit(torch.randn(256, *v['shape']))
        )
    norm['action'] = SingleFieldLinearNormalizer.create_fit(torch.randn(256, sm['action']['shape'][0]))
    policy.set_normalizer(norm)
    dev = torch.device('cuda')
    policy.eval().to(dev)
    params(policy)
    obs = make_obs(sm, dev)
    print('  wire payload:', sum(t.numel() for k, t in obs.items() if 'rgb' in k) // 1024, 'KiB of uint8 per request')
    run_suite(policy, obs)


def main() -> int:
    """Run the benchmark."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('old-ckpt')
    a.add_argument('ckpt')
    b = sub.add_parser('new-arch')
    b.add_argument('--umi-fork', default='~/hsa-ws/universal_manipulation_interface_hsa')
    args = ap.parse_args()
    print(
        f'torch {torch.__version__} on {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info()[1] / 2**20:.0f} MiB)'
    )
    if args.cmd == 'old-ckpt':
        old_ckpt(str(pathlib.Path(args.ckpt).expanduser()))
    else:
        new_arch(str(pathlib.Path(args.umi_fork).expanduser()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
