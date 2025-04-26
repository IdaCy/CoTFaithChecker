
"""Functions for loading chain‑of‑thought segment data and task hints."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator

# Default relative paths (adjust if your directory layout differs)
_DEFAULT_DATA_PATH = Path("data/mmlu")
_DEFAULT_SEG_PATH  = Path("g_cot_cluster/outputs/mmlu/DeepSeek-R1-Distill-Llama-8B")

_SEGMENTED_FILEMAP = {
    "none":                  "segmented_completions_none.json",
    "sycophancy":            "segmented_completions_sycophancy.json",
    "induced_urgency":       "segmented_completions_induced_urgency.json",
    "unethical_information": "segmented_completions_unethical_information.json",
}

def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def load_segmented_completions(hint_type: str,
                               base_path: Path | None = None) -> List[Dict]:
    """Return list of QA dicts with pre‑segmented CoTs for the given hint type."""
    if hint_type not in _SEGMENTED_FILEMAP:
        raise ValueError(f"Unknown hint_type {hint_type!r}")
    path = (base_path or _DEFAULT_SEG_PATH) / _SEGMENTED_FILEMAP[hint_type]
    return _read_json(path.resolve())

def load_hints(hint_type: str,
               base_path: Path | None = None) -> Optional[List[Dict]]:
    """Load hint metadata for a hint type (None if hint_type == 'none')."""
    if hint_type == "none":
        return None
    path = (base_path or _DEFAULT_DATA_PATH) / f"hints_{hint_type}.json"
    return _read_json(path.resolve())

def flat_segments(segmented_questions: List[Dict]) -> Iterator[Dict]:
    """Yield one dict per segment with question_id, phrase_category & text."""
    for q in segmented_questions:
        qid = q.get("question_id")
        for seg in q["segments"]:
            yield { "question_id": qid, **seg }
