"""Discover AgiBot World datasets inside a downloaded directory tree.

Two layouts are recognised (matching the HuggingFace repo structure):

ImitationLearning / RichInteraction (one ``.tar.gz`` == an episode range)::

    <root>/ImitationLearning/CommercialSpaces/task_4542/553049_553079.tar.gz
    <root>/RichInteraction/Home/task_5015/685000_694807.tar.gz

  -> after extraction the archive contains an extra ``data/`` wrapper, i.e.
     ``data/{data,meta,videos}``. The v2.1 root is that inner ``data/``.

simulation (one *variant* split into data/meta/videos multi-part archives)::

    <root>/simulation/take_cup_to_cart/g2_swift_picker/lite/data.tar.gz.000
                                                            /meta.tar.gz.000
                                                            /videos.tar.gz.000[.001...]

  -> extracted side by side they form ``{data,meta,videos}`` directly; the v2.1
     root is the extraction dir itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetItem:
    """One convertible dataset discovered on disk."""

    name: str                       # clean output name, e.g. "RI_task_4439_458713_458713"
    kind: str                       # "archive" (IL/RI) or "sim" (simulation)
    # list of archives; each archive is an ordered list of its parts
    archives: list[list[Path]] = field(default_factory=list)
    root_subdir: str = ""           # subdir holding meta/data/videos after extraction

    def __str__(self) -> str:
        n_parts = sum(len(a) for a in self.archives)
        return f"{self.name} [{self.kind}] ({len(self.archives)} archive(s), {n_parts} file(s))"


_CAT_PREFIX = {"ImitationLearning": "IL", "RichInteraction": "RI"}


def _group_multipart(files: list[Path]) -> list[Path]:
    """Sort ``foo.tar.gz.000/.001/...`` (or a single ``foo.tar.gz``) in order."""
    def key(p: Path):
        m = re.search(r"\.(\d+)$", p.name)
        return int(m.group(1)) if m else -1
    return sorted(files, key=key)


def discover_datasets(input_dir: str | Path) -> list[DatasetItem]:
    """Scan ``input_dir`` and return every dataset that can be converted."""
    root = Path(input_dir)
    items: list[DatasetItem] = []

    # --- ImitationLearning / RichInteraction: each *.tar.gz is one dataset ----
    for cat, prefix in _CAT_PREFIX.items():
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        for tgz in sorted(cat_dir.rglob("*.tar.gz")):
            task = next((p for p in tgz.parts if p.startswith("task_")), "task")
            rng = tgz.name[: -len(".tar.gz")]
            items.append(
                DatasetItem(
                    name=f"{prefix}_{task}_{rng}",
                    kind="archive",
                    archives=[[tgz]],
                    root_subdir="data",
                )
            )

    # --- simulation: each variant dir (lite / lite_depth_patch) is one dataset -
    sim_dir = root / "simulation"
    if sim_dir.is_dir():
        for variant_dir in sorted(sim_dir.glob("*/*/*")):
            if not variant_dir.is_dir():
                continue
            # collect data/meta/videos multi-part sets in this variant dir
            sets: list[list[Path]] = []
            for kind in ("data", "meta", "videos"):
                parts = list(variant_dir.glob(f"{kind}.tar.gz*"))
                if parts:
                    sets.append(_group_multipart(parts))
            if not sets:
                continue
            task = variant_dir.parts[-3]      # e.g. take_cup_to_cart
            variant = variant_dir.parts[-1]   # e.g. lite
            suffix = "" if variant == "lite" else f"_{variant}"
            items.append(
                DatasetItem(
                    name=f"SIM_{task}{suffix}",
                    kind="sim",
                    archives=sets,
                    root_subdir="",
                )
            )

    return items
