"""Read TRUE 16-bit ``head_depth`` frames from a converted dataset.

Why this exists: ``head_depth`` is stored as a 16-bit single-channel
(``gray16be``) video. lerobot's standard image path decodes it through
torchcodec as an 8-bit RGB tensor, which **quantises the depth to ~8 bits and
normalises to [0, 1]** — fine for a quick look, wrong for metric depth.

This module decodes the depth video losslessly to ``uint16`` via ffmpeg.
"""

from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import numpy as np


def find_depth_video(dataset_dir: str | Path, camera: str = "head_depth") -> Path:
    """Locate the depth mp4 inside a converted (v3.0) dataset directory."""
    hits = glob.glob(
        str(Path(dataset_dir) / "videos" / f"*{camera}*" / "**" / "*.mp4"),
        recursive=True,
    )
    if not hits:
        raise FileNotFoundError(f"no '{camera}' video under {dataset_dir}")
    return Path(sorted(hits)[0])


def _video_size(mp4: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    return int(st["width"]), int(st["height"])


def _decode_uint16(mp4: Path) -> np.ndarray:
    """Decode a ``gray16`` mp4 losslessly to ``(num_frames, H, W)`` ``uint16``."""
    w, h = _video_size(mp4)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-f", "rawvideo", "-pix_fmt", "gray16le", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<u2").reshape(-1, h, w)


def read_depth_uint16(dataset_dir: str | Path, camera: str = "head_depth") -> np.ndarray:
    """Raw 16-bit depth of the **first** depth video file (== episode 0).

    For a specific episode use :func:`read_episode_depth` (handles per-episode
    video files in multi-episode datasets).
    """
    return _decode_uint16(find_depth_video(dataset_dir, camera))


def read_episode_depth(
    dataset_dir: str | Path, episode: int = 0, camera: str = "head_depth"
) -> np.ndarray:
    """Raw 16-bit depth frames for one episode, aligned to its timeline.

    Returns ``(length, H, W)`` ``uint16`` in frame order, so ``depth[i]`` matches
    the i-th frame of that episode in ``LeRobotDataset``.
    """
    import glob
    import pandas as pd

    dataset_dir = Path(dataset_dir)
    key = camera if camera.startswith("observation.") else f"observation.images.{camera}"
    base = f"videos/{key}"

    metas = sorted(glob.glob(str(dataset_dir / "meta" / "episodes" / "**" / "*.parquet"),
                             recursive=True))
    df = pd.concat([pd.read_parquet(m) for m in metas], ignore_index=True)
    row = df[df["episode_index"] == episode]
    if row.empty:
        raise IndexError(f"episode {episode} not found in {dataset_dir}")
    row = row.iloc[0]

    ci = int(row[f"{base}/chunk_index"])
    fi = int(row[f"{base}/file_index"])
    from_ts = float(row[f"{base}/from_timestamp"])
    length = int(row["length"]) if "length" in row else \
        int(row["dataset_to_index"] - row["dataset_from_index"])
    fps = json.loads((dataset_dir / "meta" / "info.json").read_text())["fps"]

    mp4 = dataset_dir / "videos" / key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
    full = _decode_uint16(mp4)
    off = round(from_ts * fps)
    return full[off:off + length]


def colorize_depth(depth: np.ndarray, lo_hi=(2, 98), cmap: str = "turbo") -> np.ndarray:
    """Map a ``uint16`` depth frame to an ``(H, W, 3)`` ``uint8`` RGB image.

    Robust percentile normalisation (ignores zeros = invalid) + a colormap so the
    scene is actually visible (raw values sit in the bottom few % of 0..65535).
    """
    import matplotlib

    d = depth.astype(np.float32)
    valid = d[d > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(valid, lo_hi)
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    norm[d == 0] = 0
    rgb = matplotlib.colormaps[cmap](norm)[..., :3]
    return (rgb * 255).astype(np.uint8)


if __name__ == "__main__":  # tiny CLI: dump frame 0 as a viewable PNG
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser(description="Extract a viewable depth PNG.")
    ap.add_argument("dataset_dir", help="converted v3.0 dataset directory")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="depth_frame.png")
    a = ap.parse_args()

    depth = read_depth_uint16(a.dataset_dir)
    print(f"depth {depth.shape} uint16  min={depth.min()} max={depth.max()}")
    Image.fromarray(colorize_depth(depth[a.frame])).save(a.out)
    print(f"wrote {a.out}")
