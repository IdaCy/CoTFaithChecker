import json
from pathlib import Path
from typing import Dict, List, Tuple
import re
from e_confirm_xy_yx.main.logging_utils import init_logger

__all__ = [
    "get_dataset_files",
    "load_dataset",
]

logger = init_logger(log_dir=Path.cwd() / "logs", script_name="data_loader")


# ────────── cluster detection helpers ──────────
CLUSTER_PATTERNS = {
    #"arts":   re.compile(r"^wm-(book)"),
    "arts":   re.compile(r"^wm-(movie|nyt|nyc|person|song)"),
    "us":     re.compile(r"^wm-us-"),
    "world":  re.compile(r"^wm-world-"),
}

def detect_cluster(file_stem: str) -> str:
    """
    Map a dataset file-stem to one of the four clusters:
      • arts   • us   • world   • no_wm
    """
    if not file_stem.startswith("wm"):
        return "no_wm"
    for name, pat in CLUSTER_PATTERNS.items():
        if pat.match(file_stem):
            return name
    # fall-back – anything 'wm-' but not caught above
    return "world"


def get_dataset_files(
    questions_root: Path,
    dataset_folders: List[str],
    clusters: List[str] | None = None,
) -> List[Path]:
    """
    Return every *.json file inside the requested folder names.
    Example: dataset_folders = ["gt_NO_1", "gt_YES_1"].
    """
    files: List[Path] = []
    for folder in dataset_folders:
        folder_path = questions_root / folder
        current_files = sorted(folder_path.glob("*.json"))
        # optional cluster-level filtering
        if clusters is not None:
            current_files = [
                f for f in current_files
                if detect_cluster(f.stem) in clusters
            ]
            logger.info(
                f"→ kept {len(current_files)} after cluster filter {clusters}"
            )
        logger.info(f"Found {len(current_files)} files in {folder_path}")
        files.extend(current_files)
    logger.info(f"Total files collected: {len(files)}")
    return files


def load_dataset(json_path: Path) -> Dict:
    """
    Load a single JSON dataset file and return it as a dict.
    """
    logger.debug(f"Loading {json_path}")
    with json_path.open() as f:
        data = json.load(f)
    return data

__all__ = [
    "get_dataset_files",
    "load_dataset",
    "detect_cluster",
]
