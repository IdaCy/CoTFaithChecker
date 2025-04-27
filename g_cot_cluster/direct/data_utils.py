"""
data_utils.py
Utility functions to load and structure segmented chain‑of‑thought data.
"""
from __future__ import annotations
import json
import pathlib
from typing import List, Dict
from collections import Counter
import pandas as pd
import re

CATEGORY_ORDER: List[str] = [
    "problem_restating",
    "knowledge_recall",
    "concept_definition",
    "quantitative_calculation",
    "logical_deduction",
    "option_elimination",
    "assumption_validation",
    "uncertainty_expression",
    "self_questioning",
    "backtracking_revision",
    "decision_confirmation",
    "answer_reporting",
]


def load_segmented_directory(directory: str | pathlib.Path) -> pd.DataFrame:
    """Load every *segmented_completions_*.json* file inside *directory* and
    return a flat :pyclass:`pandas.DataFrame` with one row per segment.

    Columns: ``question_id``, ``hint_type``, ``segment_idx``, ``phrase_category``,
    ``text``, ``start``, ``end``.
    """
    directory = pathlib.Path(directory)
    rows = []
    pattern = "segmented_completions_*.json"
    prefix = "segmented_completions_"
    
    for path in directory.glob(pattern):
        hint_type = path.stem[len(prefix):]  # e.g. *sycophancy* / *none*
        with path.open() as fh:
            data = json.load(fh)
        for q in data:
            qid = q["question_id"]
            for idx, seg in enumerate(q["segments"]):
                rows.append(
                    {
                        "question_id": qid,
                        "hint_type": hint_type,
                        "segment_idx": idx,
                        "phrase_category": seg["phrase_category"],
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                    }
                )
    df = pd.DataFrame(rows)
    df["phrase_category"] = pd.Categorical(
        df["phrase_category"], categories=CATEGORY_ORDER, ordered=True
    )
    return df


def sequence_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the segment‑level dataframe into one row per *question_id* &
    *hint_type*, adding ordered lists of categories and the concatenated text.
    """
    seq_df = (
        df.sort_values(["question_id", "segment_idx"])
        .groupby(["question_id", "hint_type"])
        .agg(
            category_sequence=("phrase_category", list),
            full_text=("text", lambda x: "\n".join(x)),
        )
        .reset_index()
    )
    return seq_df


ANSWER_PATTERN = re.compile(r"\*\*Answer:\*\*\s*\[?\s*([A-Za-z0-9]+)\s*\]?", re.IGNORECASE)

def extract_predicted_answer(text: str) -> str | None:
    """Extract the reported answer (e.g. ``"A"`` or ``"42"``) from a segment
    containing the typical pattern ``**Answer:** [ D ]``.
    Returns *None* if no answer is found.
    """
    m = ANSWER_PATTERN.search(text)
    return m.group(1).upper() if m else None

# ---------------------------------------------------------------------------
# Accuracy & “switch” information
# ---------------------------------------------------------------------------
import json, pathlib
from typing import List

from pathlib import Path
import json
import pandas as pd

def load_accuracy_logs(
    base_dir: str | Path,
    mcq_file: str | Path,
    none_log: str | Path = "answers_none.json",
):
    """
    base_dir/
        induced_urgency/          switch_analysis_with_500.json
        sycophancy/               switch_analysis_with_500.json
        misleading_justification/ switch_analysis_with_500.json
        missing_chain/            switch_analysis_with_500.json
    """
    base_dir  = Path(base_dir)
    mcq_file  = Path(mcq_file)
    none_log  = Path(none_log)

    # ------------------------------------------------------------------ #
    # map question_id → correct option
    # ------------------------------------------------------------------ #
    correct = {int(row["question_id"]): row["correct"]
               for row in json.load(mcq_file.open())}

    rows = []

    # ------------------------------------------------------------------ #
    # hinted conditions (search *recursively*)
    # ------------------------------------------------------------------ #
    for path in base_dir.rglob("switch*_*.json"):
        hint_type = path.parent.name              # folder name = hint label
        with path.open() as fh:
            for rec in json.load(fh):
                qid = int(rec["question_id"])
                rows.append(
                    dict(
                        question_id=qid,
                        hint_type=hint_type,
                        accuracy=int(rec["is_correct_option"]),
                        switched=bool(rec["switched"]),
                        to_intended_hint=bool(rec["to_intended_hint"]),
                        hint_option=rec["hint_option"],
                    )
                )

    # ------------------------------------------------------------------ #
    # no-hint baseline
    # ------------------------------------------------------------------ #
    with none_log.open() as fh:
        for rec in json.load(fh):
            qid  = int(rec["question_id"])
            pred = rec["verified_answer"]
            rows.append(
                dict(
                    question_id=qid,
                    hint_type="none",
                    accuracy=int(pred == correct[qid]),
                    switched=False,
                    to_intended_hint=False,
                    hint_option=pred,
                )
            )

    return pd.DataFrame(rows)
