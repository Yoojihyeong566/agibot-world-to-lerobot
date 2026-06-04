"""Rerun viewer that renders ``head_depth`` correctly.

``lerobot-dataset-viz`` logs every camera as ``rr.Image``, so 16-bit depth (whose
values sit in the bottom few % of the range) shows up almost black. This viewer
logs depth cameras as **``rr.DepthImage``** instead — rerun then auto-ranges and
colormaps them, so the depth is actually visible. RGB cameras are logged as
usual.

Usage::

    # spawn the rerun viewer (needs a display)
    python -m agibot2lerobot.viz_depth <dataset_dir> --episode 0

    # headless: write a .rrd to open later with `rerun file.rrd`
    python -m agibot2lerobot.viz_depth <dataset_dir> --episode 0 --save out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _to_hwc_uint8(chw_float: np.ndarray) -> np.ndarray:
    """(C,H,W) float[0,1] -> (H,W,C) uint8."""
    img = np.transpose(chw_float, (1, 2, 0))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def visualize_episode(
    dataset_dir: str | Path,
    episode: int = 0,
    namespace: str = "agibot",
    save_dir: str | Path | None = None,
    max_frames: int | None = None,
    depth_only: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    import rerun as rr
    from rerun.components import Colormap
    from torch.utils.data import DataLoader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .depth import read_episode_depth

    dataset_dir = Path(dataset_dir).resolve()
    name = dataset_dir.name
    ds = LeRobotDataset(f"{namespace}/{name}", root=str(dataset_dir), episodes=[episode])

    spawn = save_dir is None
    rr.init(f"{namespace}/{name}/episode_{episode}", spawn=spawn)

    depth_keys = [k for k in ds.meta.camera_keys if "depth" in k.lower()]
    rgb_keys = [] if depth_only else [k for k in ds.meta.camera_keys if k not in depth_keys]
    print(f"episode {episode}: {ds.num_frames} frames | depth={depth_keys} rgb={rgb_keys}")
    if spawn and ds.num_frames > 600 and not depth_only:
        print("  NOTE: streaming a long, multi-camera episode to the live viewer can\n"
              "        overload rerun (gRPC transport error). If it dies, use --save to\n"
              "        write a .rrd, add --depth-only, or just: agibot2lerobot depth ...")

    # Pre-load TRUE 16-bit depth per episode (lerobot's path squashes it to 8-bit
    # -> banding). One stable display range for the whole episode.
    depth16: dict[str, np.ndarray] = {}
    depth_range: dict[str, tuple[float, float]] = {}
    for key in depth_keys:
        d = read_episode_depth(dataset_dir, episode=episode, camera=key.split(".")[-1])
        depth16[key] = d
        valid = d[d > 0]
        lo, hi = (float(np.percentile(valid, 2)), float(np.percentile(valid, 98))) \
            if valid.size else (0.0, 1.0)
        depth_range[key] = (lo, hi)
        print(f"  {key}: 16-bit {d.shape}, {len(np.unique(d))} levels, range~[{lo:.0f},{hi:.0f}]")

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    seen = 0
    first_index = None
    for batch in loader:
        if first_index is None:
            first_index = batch["index"][0].item()
        for i in range(batch["index"].shape[0]):
            if max_frames is not None and seen >= max_frames:
                break
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            for key in rgb_keys:
                rr.log(key, rr.Image(_to_hwc_uint8(batch[key][i].numpy())))

            for key in depth_keys:
                frame = depth16[key][seen]                      # continuous uint16
                lo, hi = depth_range[key]
                rr.log(key, rr.DepthImage(
                    frame, colormap=Colormap.Grayscale, depth_range=[lo, hi]))
            seen += 1
        if max_frames is not None and seen >= max_frames:
            break

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{namespace}_{name}_episode_{episode}.rrd"
        rr.save(str(path))
        print(f"wrote {path}\n  open with:  rerun {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="converted v3.0 dataset directory")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--namespace", default="agibot")
    ap.add_argument("--save", dest="save_dir", default=None,
                    help="write a .rrd to this dir instead of spawning a viewer")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="limit number of frames (handy for a quick look)")
    ap.add_argument("--depth-only", action="store_true",
                    help="log only depth cameras (much lighter; avoids viewer overload)")
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args(argv)
    visualize_episode(
        a.dataset_dir, episode=a.episode, namespace=a.namespace,
        save_dir=a.save_dir, max_frames=a.max_frames, depth_only=a.depth_only,
        num_workers=a.num_workers,
    )


if __name__ == "__main__":
    main()
