"""Command line interface: ``python -m agibot2lerobot <command>``.

Commands
--------
list      Discover datasets under a download dir (no conversion).
convert   Extract + convert v2.1 -> v3.0 into an output dir.
viz       Print the ready-to-run lerobot-dataset-viz command for a dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .discover import discover_datasets
from .convert import convert_all


def _cmd_list(args):
    items = discover_datasets(args.input)
    if not items:
        print(f"No AgiBot datasets found under {args.input}")
        return
    print(f"Found {len(items)} dataset(s) under {args.input}:")
    for it in items:
        print(f"  - {it}")


def _cmd_convert(args):
    results = convert_all(
        input_dir=args.input,
        output_dir=args.output,
        workers=args.workers,
        only=args.only,
        keep_extracted=args.keep_extracted,
        keep_v21_backup=args.keep_v21_backup,
        repo_namespace=args.namespace,
    )
    print("\n==================== SUMMARY ====================")
    for name, status in results.items():
        print(f"  {status:8s}  {name}" if status == "ok" else f"  {name}: {status}")
    failed = [n for n, s in results.items() if s != "ok"]
    raise SystemExit(1 if failed else 0)


def _cmd_depth(args):
    from .depth import export_grayscale_video
    out = export_grayscale_video(args.dataset, episode=args.episode, out=args.out)
    print(f"wrote {out}  (grayscale depth video — open in any player)")


def _cmd_viz(args):
    ds = Path(args.dataset).resolve()
    name = ds.name
    print("conda activate agibot_world")
    print(
        f'lerobot-dataset-viz --repo-id {args.namespace}/{name} \\\n'
        f'  --root "{ds}" \\\n'
        f'  --mode local --episode-index {args.episode} --display-compressed-images'
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="agibot2lerobot")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="discover datasets (no conversion)")
    pl.add_argument("input", help="AgiBot download dir (contains ImitationLearning/ etc.)")
    pl.set_defaults(func=_cmd_list)

    pc = sub.add_parser("convert", help="extract + convert v2.1 -> v3.0")
    pc.add_argument("input", help="AgiBot download dir")
    pc.add_argument("output", help="output dir for v3.0 datasets")
    pc.add_argument("--workers", type=int, default=8)
    pc.add_argument("--only", nargs="*", default=None,
                    help="only convert datasets whose name contains any of these")
    pc.add_argument("--namespace", default="agibot",
                    help="repo-id namespace / output subfolder (default: agibot)")
    pc.add_argument("--keep-extracted", action="store_true",
                    help="keep the intermediate extracted/ files")
    pc.add_argument("--keep-v21-backup", action="store_true",
                    help="keep the <name>_old v2.1 backup lerobot leaves behind")
    pc.set_defaults(func=_cmd_convert)

    pd = sub.add_parser("depth", help="export head_depth as a grayscale mp4 (simple, robust)")
    pd.add_argument("dataset", help="path to a converted v3.0 dataset dir")
    pd.add_argument("--episode", type=int, default=0)
    pd.add_argument("--out", default="depth.mp4")
    pd.set_defaults(func=_cmd_depth)

    pv = sub.add_parser("viz", help="print the viz command for a converted dataset")
    pv.add_argument("dataset", help="path to a converted v3.0 dataset dir")
    pv.add_argument("--episode", type=int, default=0)
    pv.add_argument("--namespace", default="agibot")
    pv.set_defaults(func=_cmd_viz)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
