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
    "spec":   re.compile(r"^wm-(nyc|person|song|world-structure)"),
    "arts":   re.compile(r"^wm-(book|movie|nyt|nyc|person|song)"),
    "thing":   re.compile(r"^wm-(book|movie|nyt)"),
    "human":   re.compile(r"^wm-(nyc|person|song)"),
    "us":     re.compile(r"^wm-us-"),
    "world":  re.compile(r"^wm-world-"),
    "us-and-world":  re.compile(r"^wm-(us|world)-"),
    "us_spec":  re.compile(r"^wm-(wm-us-county-lat|wm-us-county-long|wm-us-county-popu|wm-us-natural-lat|wm-us-natural-long|wm-us-structure|wm-us-zip|wm-world-structure)-"),
}
"""
not - already done:
wm-us-city-dens_gt_YES_1_60629297_DeepSeek-R1-Distill-Llama-8B_completions
wm-us-city-lat_gt_YES_1_2a33a48d_DeepSeek-R1-Distill-Llama-8B_completions
wm-us-city-long_gt_YES_1_45faecb7_DeepSeek-R1-Distill-Llama-8B_completions.json
wm-us-city-popu_gt_YES_1_8af18eca_DeepSeek-R1-Distill-Llama-8B_completions
wm-us-college-lat_gt_YES_1_2f416a49_DeepSeek-R1-Distill-Llama-8B_completions
wm-us-college-long_gt_YES_1_5d3e577e_DeepSeek-R1-Distill-Llama-8B_completions.json
"""

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
