#!/usr/bin/env python
"""merge_outputs.py

Merges the **raw generations** and **analysis** JSON files produced by two
separate runs of `temp_robust_new_refactor.py` into a single set of files
with a `5001` suffix.

Assumptions
-----------
- The two runs used the *same* model/dataset/hint-type paths and therefore
  live under the same directory structure:

    f_temp_check/outputs/<dataset>/<model_suffix>/<hint_type>/

- The filenames follow the exact pattern produced by the main script:
    * temp_generations_raw_<dataset>_<N>.json
    * temp_analysis_details_<N>.json
    * temp_analysis_summary_<N>.json

- The two source runs do **not** overlap in `question_id`s.

Usage (from repository root)
---------------------------
    python f_temp_check/merge_outputs.py \
        --dataset mmlu \
        --model-suffix DeepSeek-R1-Distill-Llama-8B \
        --hint-type sycophancy \
        --first 2001 \
        --second 3000

This will create *_5001.json* files alongside the originals.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Wrote {path} ({path.stat().st_size/1024:.1f} KB)")


def merge_lists(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Concatenate lists and sanity-check duplicate question_ids."""
    combined: List[Dict[str, Any]] = []
    seen: set[Any] = set()
    for lst in lists:
        for item in lst:
            qid = item.get("question_id")
            if qid in seen:
                logging.warning(f"Duplicate question_id found while merging: {qid}. Skipping.")
                continue
            seen.add(qid)
            combined.append(item)
    # Sort by integer question_id when possible, fallback to original order
    try:
        combined.sort(key=lambda x: int(x.get("question_id")))
    except Exception:
        pass
    return combined

# -----------------------------------------------------------------------------
# Main merge logic
# -----------------------------------------------------------------------------

def merge_outputs(base_dir: Path, dataset: str, first: int, second: int):
    """Merge raw generation, details, and summary files for two runs."""
    total = first + second  # 2001 + 3000 = 5001 (as per user scenario)

    def gen_file(name_pattern: str, n: int) -> Path:
        return base_dir / (name_pattern.format(dataset=dataset, n=n))

    # --- 1. Merge *raw generations* files ------------------------------------
    raw_pattern = "temp_generations_raw_{dataset}_{n}.json"
    logging.info("Merging raw generation files…")
    raw_first = load_json(gen_file(raw_pattern, first))
    raw_second = load_json(gen_file(raw_pattern, second))

    merged_raw = {
        "config": {
            "merged_from": [first, second],
            "n_questions": total
        },
        "raw_generations": merge_lists(raw_first.get("raw_generations", []),
                                        raw_second.get("raw_generations", []))
    }
    save_json(merged_raw, gen_file(raw_pattern, total))

    # --- 2. Merge *analysis details* files -----------------------------------
    details_pattern = "temp_analysis_details_{n}.json"
    logging.info("Merging analysis *details* files…")
    det_first = load_json(gen_file(details_pattern, first))
    det_second = load_json(gen_file(details_pattern, second))
    merged_details = {
        "config": {
            "merged_from": [first, second],
            "n_questions": total
        },
        "detailed_analysis": merge_lists(det_first.get("detailed_analysis", []),
                                          det_second.get("detailed_analysis", []))
    }
    save_json(merged_details, gen_file(details_pattern, total))

    # --- 3. Merge *analysis summary* files -----------------------------------
    summary_pattern = "temp_analysis_summary_{n}.json"
    logging.info("Merging analysis *summary* files…")
    sum_first = load_json(gen_file(summary_pattern, first))
    sum_second = load_json(gen_file(summary_pattern, second))

    merged_summary = {
        "config": {
            "merged_from": [first, second],
            "n_questions": total
        },
        "results_per_question_summary": merge_lists(sum_first.get("results_per_question_summary", []),
                                                    sum_second.get("results_per_question_summary", [])),
        "overall_summary": {
            "placeholder": "Overall summary not recomputed after merge."
        }
    }
    save_json(merged_summary, gen_file(summary_pattern, total))

    logging.info("Merging complete!")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge two output runs into a combined set with 5001 suffix.")
    parser.add_argument("--dataset", default="mmlu", help="Dataset name (default: mmlu)")
    parser.add_argument("--model-suffix", default="DeepSeek-R1-Distill-Llama-8B", help="Model directory suffix (e.g., DeepSeek-R1-Distill-Llama-8B)")
    parser.add_argument("--hint-type", default="sycophancy", help="Hint type sub-directory (default: sycophancy)")
    parser.add_argument("--first", type=int, default=2001, help="First run question count (default: 2001)")
    parser.add_argument("--second", type=int, default=3000, help="Second run question count (default: 3000)")

    args = parser.parse_args()

    base_dir = Path("f_temp_check") / "outputs" / args.dataset / args.model_suffix / args.hint_type
    if not base_dir.is_dir():
        parser.error(f"Base directory not found: {base_dir}")

    merge_outputs(base_dir, args.dataset, args.first, args.second)

if __name__ == "__main__":
    main() 