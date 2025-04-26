
"""Utility functions for loading segmented CoT JSON files."""
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List

__all__ = ["load_segmented_completions", "iter_segments"]

def load_segmented_completions(file_path: str) -> List[Dict[str, Any]]:
    "Load a completions file from JSON."
    path = Path(file_path)
    with path.open() as f:
        data = json.load(f)
    return data

def iter_segments(data: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    "Yield a flat stream of segment dicts with question_id attached."
    for q in data:
        qid = q.get("question_id")
        for seg in q["segments"]:
            seg_out = dict(seg)
            seg_out["question_id"] = qid
            yield seg_out
