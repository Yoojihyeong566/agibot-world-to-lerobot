"""agibot2lerobot — Convert AgiBot World 2026 downloads into LeRobot v3.0 datasets.

AgiBot World 2026 ships datasets in **LeRobot v2.1** format, packed as `.tar.gz`
archives (ImitationLearning / RichInteraction) or multi-part archives
(simulation). The current `lerobot` (>=0.5) only reads **v3.0**, so each dataset
must be (1) extracted and (2) version-converted v2.1 -> v3.0.

This package automates discovery + extraction + conversion. See README.md.
"""

__version__ = "0.1.0"

from .extract import extract_archive
from .discover import discover_datasets, DatasetItem
from .convert import convert_one, convert_all

__all__ = [
    "extract_archive",
    "discover_datasets",
    "DatasetItem",
    "convert_one",
    "convert_all",
]
