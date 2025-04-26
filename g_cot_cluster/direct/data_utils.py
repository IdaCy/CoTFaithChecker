"""Utilities for loading and preprocessing segmented CoT files."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional

import pandas as pd

# Regular expression that tries to pull the model's predicted answer
_RE_BRACKET = re.compile(r"\[\s*([A-Za-z0-9]+)\s*\]")

_HINT_REGEX = re.compile(r"segmented_completions_(.+?)\.json$")


################################################################################
# Loading helpers
################################################################################

def _extract_hint_type(path: Path) -> str:
    m = _HINT_REGEX.search(path.name)
    return m.group(1) if m else "unknown"


def _extract_predicted_answer(segments: Sequence[Dict[str, Any]]) -> Optional[str]:
    for s in segments[::-1]:
        if s.get("phrase_category") == "answer_reporting":
            text = s.get("text", "")
            m = _RE_BRACKET.search(text)
            return m.group(1) if m else text.strip()
    return None


def _has_backtracking(segments: Sequence[Dict[str, Any]]) -> bool:
    return any(s.get("phrase_category") == "backtracking_revision" for s in segments)


def load_data(root_dir: str | Path) -> pd.DataFrame:
    """Load **all** `segmented_completions_*.json` files from *root_dir*.

    Returns
    -------
    DataFrame  with one row per question and columns:
        question_id, hint_type, segments, category_sequence,
        full_text, has_backtracking, predicted_answer, is_correct (nullable)
    """
    root = Path(root_dir)
    rows = []
    for fp in root.glob("segmented_completions_*.json"):
        hint_type = _extract_hint_type(fp)
        with fp.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for q in data:
            segments: List[Dict[str, Any]] = q["segments"]
            cat_seq = [s["phrase_category"] for s in segments]
            rows.append(
                {
                    "question_id": q.get("question_id"),
                    "hint_type": hint_type,
                    "segments": segments,
                    "category_sequence": cat_seq,
                    "full_text": " ".join(s["text"] for s in segments),
                    "has_backtracking": _has_backtracking(segments),
                    "predicted_answer": _extract_predicted_answer(segments),
                    "is_correct": q.get("is_correct"),  # may be None
                }
            )
    df = pd.DataFrame(rows)
    return df


def merge_accuracy(df: pd.DataFrame, answer_key: pd.DataFrame) -> pd.DataFrame:
    """Attach ground‑truth correctness.

    *answer_key* must have columns [question_id, correct_answer].
    """
    answer_key = answer_key.rename(columns={"correct_answer": "_gt"})
    out = df.merge(answer_key, on="question_id", how="left")
    out["is_correct"] = out["is_correct"].fillna(
        out.apply(lambda r: r["predicted_answer"] == r["_gt"], axis=1)
    )
    out = out.drop(columns="_gt")
    return out
