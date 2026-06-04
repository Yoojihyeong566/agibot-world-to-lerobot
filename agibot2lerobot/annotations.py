"""Preserve AgiBot language / annotation layers across the v2.1 -> v3.0 conversion.

`convert_dataset_v21_to_v30` drops AgiBot's custom `info.json` fields ("Unknown
fields ... ignored"), so the v3.0 dataset keeps only the one-line task label.
The rich language annotations live ONLY in the original v2.1 `meta/info.json`:

  * ``instruction_segments``  — per-episode, per-step skill + NL instruction
  * ``key_frame``            — Task Frame (subtask NL) + 2D Bounding Box (objects)
  * ``high_level_instruction`` — per-episode high-level NL goal (simulation)
  * ``take_over``            — teleop takeover spans

We copy these (plus the raw ``meta/annotations.json``) into a sidecar
``meta/agibot_annotations.json`` in the converted dataset so nothing is lost.
"""

from __future__ import annotations

import json
from pathlib import Path

# info.json keys carrying language / annotation info that conversion discards
LANGUAGE_FIELDS = [
    "instruction_segments",
    "key_frame",
    "high_level_instruction",
    "take_over",
]

SIDECAR_NAME = "agibot_annotations.json"


def extract_annotations(v21_meta_dir: str | Path) -> dict:
    """Pull the language/annotation layers out of a v2.1 ``meta/`` directory."""
    meta = Path(v21_meta_dir)
    info = json.loads((meta / "info.json").read_text())

    out: dict = {
        "source_codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "fields": {k: info[k] for k in LANGUAGE_FIELDS if k in info},
        "present": [k for k in LANGUAGE_FIELDS if k in info],
    }
    # also stash the raw annotations.json (episode_id/task_id/source paths) if any
    raw = meta / "annotations.json"
    if raw.is_file():
        try:
            out["annotations_json"] = json.loads(raw.read_text())
        except Exception:
            pass
    return out


def save_annotations(annos: dict, out_meta_dir: str | Path) -> Path:
    """Write the preserved annotations next to the converted ``meta/``."""
    path = Path(out_meta_dir) / SIDECAR_NAME
    path.write_text(json.dumps(annos, ensure_ascii=False, indent=1))
    return path


def load_annotations(dataset_dir: str | Path) -> dict:
    """Load the preserved annotations from a converted dataset (``meta/`` sidecar)."""
    path = Path(dataset_dir) / "meta" / SIDECAR_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no {SIDECAR_NAME} in {dataset_dir} — was it converted with annotation "
            "preservation enabled?"
        )
    return json.loads(path.read_text())


def instruction_segments(dataset_dir: str | Path, episode: int):
    """Convenience: step-level instruction segments for one episode (or [])."""
    fields = load_annotations(dataset_dir).get("fields", {})
    return fields.get("instruction_segments", {}).get(str(episode), [])
