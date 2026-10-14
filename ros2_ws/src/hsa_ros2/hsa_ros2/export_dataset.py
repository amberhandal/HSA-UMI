"""
Inspect a recorded dataset and/or pack it into the ``.zarr.zip`` the training script takes.

ROS-free, so it runs anywhere zarr is installed (the ROS interpreter, or the analysis venv):

    ros2 run hsa_ros2 export_dataset ~/hsa_data/peg_v1                   # summary only
    ros2 run hsa_ros2 export_dataset ~/hsa_data/peg_v1 --episodes        # per-episode ledger from the sidecars
    ros2 run hsa_ros2 export_dataset ~/hsa_data/peg_v1 --zip             # -> ~/hsa_data/peg_v1/peg_v1.zarr.zip
    ros2 run hsa_ros2 export_dataset ~/hsa_data/peg_v1 --zip --include-failures   # keep failed episodes too
    ros2 run hsa_ros2 export_dataset ~/hsa_data/peg_v1 --zip --stage turn -o turn.zarr.zip   # one subtask only
    python3 -m hsa_ros2.export_dataset ~/hsa_data/peg_v1 --zip -o /data/peg_v1.zarr.zip
    python3 -m hsa_ros2.export_dataset --check /data/peg_v1.zarr.zip     # reopen a zip the way UmiDataset does

Episodes saved as failures (teleop key f) are left out of the zip unless --include-failures is given.
The recorder's own `/hsa_recorder/export_zip` service does the same as `--zip` while it runs.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from hsa_ros2 import dataset_writer as dw


def _stage_names(d: pathlib.Path) -> list[str]:
    """Stage names from the newest sidecar that carries them, else the defaults."""
    import json

    for f in sorted((d / 'episodes').glob('episode_*.json'), reverse=True):
        names = json.loads(f.read_text()).get('stage_names')
        if names:
            return list(names)
    return list(dw.DEFAULT_STAGE_NAMES)


def list_episodes(d: pathlib.Path) -> int:
    """One line per sidecar, with every plan column as its own column: the dataset's own ledger."""
    import json

    files = sorted((d / 'episodes').glob('episode_*.json'))
    if not files:
        print(f'no sidecars under {d / "episodes"}')
        return 1
    recs = [json.loads(f.read_text()) for f in files]
    cols = sorted({k for r in recs for k in (r.get('plan') or {})})
    head = (
        f'{"ep":>4} {"outcome":8s} {"steps":>5} {"dur s":>6}  {"condition":16s} ' + ' '.join(f'{c:12s}' for c in cols) + ' wrist_set'
    )
    print(head)
    for r in recs:
        plan = r.get('plan') or {}
        print(
            f'{r["episode"]:4d} {r.get("outcome", "success"):8s} {r["n_steps"]:5d} {r["duration_s"]:6.1f}  {str(r.get("condition", ""))[:16]:16s} '
            + ' '.join(f'{str(plan.get(c, ""))[:12]:12s}' for c in cols)
            + f' {r.get("wrist_set", "")}'
            + ''.join(f'  {st.get("name") or st["stage"]} {st["duration_s"]:.1f}s' for st in r.get('stages') or [])
        )
    return 0


def main(argv=None) -> int:
    """Run the export command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dir', nargs='?', help='recorder output dir (contains replay_buffer.zarr)')
    ap.add_argument('--zip', action='store_true', help='pack into a .zarr.zip')
    ap.add_argument('-o', '--out', default=None, help='zip path (default: <output_dir>/<name>.zarr.zip)')
    ap.add_argument(
        '--stage',
        help='with --zip: export only this subtask (index or name, e.g. insert / turn / open); each run of it '
        'becomes its own episode',
    )
    ap.add_argument('--min-steps', type=int, default=2, help='with --stage: skip shorter segments')
    ap.add_argument(
        '--include-failures', action='store_true', help='with --zip: keep episodes saved as failures'
    )
    ap.add_argument('--check', metavar='ZIP', help='reopen an exported .zarr.zip via zarr.ZipStore and summarise it')
    ap.add_argument(
        '--episodes', action='store_true', help='list the per-episode sidecars: steps, condition, plan columns'
    )
    args = ap.parse_args(argv)

    if args.check:
        print(dw.open_zip_summary(args.check))
        return 0
    if not args.output_dir:
        ap.error('output_dir is required unless --check is given')
    d = pathlib.Path(args.output_dir).expanduser()
    if not (d / 'replay_buffer.zarr').exists():
        print(f'no replay_buffer.zarr under {d}', file=sys.stderr)
        return 1
    if args.episodes:
        return list_episodes(d)
    w = dw.ReplayBufferWriter(d)
    print(w.summary())
    if args.zip:
        outcomes = w.episode_outcomes()
        keep = [i for i, o in enumerate(outcomes) if args.include_failures or o != 'failure']
        print(f'{outcomes.count("success")} success / {outcomes.count("failure")} failure; exporting {len(keep)}')
        stage = None
        if args.stage is not None:
            stage = dw.stage_index(args.stage, _stage_names(d))
            print(f'subtask: stage {stage} ({args.stage})')
        out = w.export_zip(args.out, episodes=keep, stage=stage, min_steps=args.min_steps)
        print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB)')
        print(dw.open_zip_summary(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
