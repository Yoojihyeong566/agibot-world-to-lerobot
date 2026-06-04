"""Extract + convert AgiBot v2.1 datasets to LeRobot v3.0.

The actual version conversion is delegated to lerobot's own
``convert_dataset_v21_to_v30.convert_dataset`` (do not reinvent it). We only
handle extraction, locating the v2.1 root, output layout and cleanup.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .annotations import extract_annotations, save_annotations
from .discover import DatasetItem, discover_datasets
from .extract import extract_archive


def _load_codebase_version(meta_root: Path) -> str | None:
    info = meta_root / "meta" / "info.json"
    if not info.is_file():
        return None
    try:
        return json.loads(info.read_text())["codebase_version"]
    except Exception:
        return None


def convert_one(
    item: DatasetItem,
    output_dir: str | Path,
    workers: int = 8,
    extract_dir: str | Path | None = None,
    keep_extracted: bool = False,
    keep_v21_backup: bool = False,
    repo_namespace: str = "agibot",
    preserve_annotations: bool = True,
) -> Path:
    """Extract and convert a single dataset. Returns the v3.0 dataset path.

    Layout produced::

        <output_dir>/<namespace>/<item.name>/{meta,data,videos}   # v3.0
    """
    # lerobot import is deferred so `--help`/discovery work without it installed
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    output_dir = Path(output_dir)
    out = output_dir / repo_namespace / item.name
    extract_root = Path(extract_dir) if extract_dir else output_dir / "_extracted"
    work = extract_root / item.name

    # clean any previous attempt
    for p in (out, out.parent / f"{item.name}_old", out.parent / f"{item.name}_v30", work):
        if p.exists():
            shutil.rmtree(p)

    # 1) extract every archive into the same working dir
    work.mkdir(parents=True, exist_ok=True)
    for parts in item.archives:
        extract_archive(parts, work)

    # 2) locate the v2.1 dataset root (meta/data/videos live here)
    v21_root = work / item.root_subdir if item.root_subdir else work
    ver = _load_codebase_version(v21_root)
    if ver is None:
        raise FileNotFoundError(f"no meta/info.json under {v21_root}")
    if ver != "v2.1":
        raise ValueError(f"{item.name}: expected codebase_version v2.1, got {ver}")

    # grab the language/annotation layers BEFORE conversion strips them
    annos = extract_annotations(v21_root / "meta") if preserve_annotations else None

    # 3) copy to output, then convert in place (lerobot leaves a *_old backup)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(v21_root, out)
    convert_dataset(
        repo_id=f"{repo_namespace}/{item.name}",
        root=str(out),
        push_to_hub=False,
    )

    # 3b) restore the stripped annotations as a meta/ sidecar
    if annos is not None:
        save_annotations(annos, out / "meta")

    # 4) cleanup
    backup = out.parent / f"{item.name}_old"
    if backup.exists() and not keep_v21_backup:
        shutil.rmtree(backup)
    if not keep_extracted:
        shutil.rmtree(work, ignore_errors=True)

    return out


def convert_all(
    input_dir: str | Path,
    output_dir: str | Path,
    workers: int = 8,
    only: list[str] | None = None,
    **kwargs,
) -> dict[str, str]:
    """Discover and convert every dataset under ``input_dir``.

    Args:
        only: optional list of dataset name substrings to restrict to.

    Returns:
        ``{name: "ok" | "ERROR: ..."}`` per dataset.
    """
    items = discover_datasets(input_dir)
    if only:
        items = [it for it in items if any(s in it.name for s in only)]

    results: dict[str, str] = {}
    for it in items:
        print(f"\n=== {it} ===", flush=True)
        try:
            out = convert_one(it, output_dir, workers=workers, **kwargs)
            n = _load_codebase_version(out)
            results[it.name] = "ok"
            print(f"--> {out}  (version={n})", flush=True)
        except Exception as e:  # keep going on failure
            results[it.name] = f"ERROR: {e}"
            print(f"!!! {it.name} failed: {e}", flush=True)
    return results
