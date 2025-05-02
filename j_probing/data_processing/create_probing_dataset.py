import os
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional

import importlib.util

# --------------------------------------------------------------------------------------
# Load shared experiment metadata from `data_prep.py`
# --------------------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent

# spec = importlib.util.spec_from_file_location("data_prep", THIS_DIR / "data_prep.py")
# data_prep = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(data_prep)  # type: ignore

# These variables are defined in j_probing/data_processing/data_prep.py
DATASET_NAME = "mmlu"  # e.g. "mmlu"
MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"       # e.g. "DeepSeek-R1-Distill-Llama-8B"
HINT_TYPE = "sycophancy"         # e.g. "sycophancy"
N_QUESTIONS = 301     # e.g. 301

# --------------------------------------------------------------------------------------
# Input paths
# --------------------------------------------------------------------------------------
BASE_INPUT_DIR = Path("f_temp_check") / "outputs" / DATASET_NAME / MODEL_NAME / HINT_TYPE
ANALYSIS_FILE = BASE_INPUT_DIR / f"temp_analysis_summary_{N_QUESTIONS}.json"
RAW_GENS_FILE = BASE_INPUT_DIR / f"temp_generations_raw_{DATASET_NAME}_{N_QUESTIONS}.json"

# --------------------------------------------------------------------------------------
# Output path (create if necessary)
# --------------------------------------------------------------------------------------
OUTPUT_DIR = (
    Path("j_probing")
    / "data"
    / DATASET_NAME
    / MODEL_NAME
    / HINT_TYPE
    / str(N_QUESTIONS)
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "probing_data.json"


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
THINK_DELIM = "<think>\n"  # delimiter that separates prompt from assistant reasoning


def _compute_probs(agg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Compute prob_verb_agg and prob_verb_match from aggregated_counts."""
    num_analyzed = agg["num_generations_analyzed_for_verbalization"]
    match_hint = agg["match_hint_count"]

    prob_verb_agg = (
        agg["verbalize_hint_count"] / num_analyzed if num_analyzed else None
    )
    prob_verb_match = (
        agg["match_and_verbalize_count"] / match_hint if match_hint else None
    )
    return {
        "prob_verb_agg": prob_verb_agg,
        "prob_verb_match": prob_verb_match,
    }


def _extract_prompt(gen_str: str) -> str:
    """Return the prompt portion up to and including the last `<think>\n`."""
    idx = gen_str.rfind(THINK_DELIM)
    if idx == -1:
        # Fallback: return entire string if delimiter missing
        return gen_str
    return gen_str[: idx + len(THINK_DELIM)]


def _token_positions(prompt: str) -> List[int]:
    """Derive token positions [assistant_idx, think_idx, hint_idx] using simple whitespace tokenisation.

    assistant_idx  : last token in the prompt
    think_idx      : second-to-last token (expected to be '<think>')
    hint_idx       : first occurrence of a token that contains the substring 'hint' (case-insensitive)
                     If not found, returns -1 for that entry.
    """
    tokens = prompt.strip().split()
    if not tokens:
        return [-1, -1, -1]

    assistant_idx = len(tokens) -1
    think_idx = len(tokens)

    # Search for the hint token (case-insensitive search)
    hint_idx = next(
        (i for i, tok in enumerate(tokens) if "hint" in tok.lower()),
        -1,
    )
    return [-1, -2, -3]


# --------------------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------------------

def main() -> None:
    # 1. Load analysis summary
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    # Build a mapping from question_id to computed stats
    summary_map: Dict[int, Dict[str, Any]] = {}
    for entry in analysis_data["results_per_question_summary"]:
        qid = entry["question_id"]
        stats = {
            "original_verbalizes_hint": entry["original_verbalizes_hint"],
            **_compute_probs(entry["aggregated_counts"]),
        }
        summary_map[qid] = stats

    # 2. Load raw generations (potentially large file)
    with open(RAW_GENS_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    probing_records: List[Dict[str, Any]] = []

    for q in raw_data["raw_generations"]:
        qid: int = q["question_id"]
        if qid not in summary_map:
            # Skip if not present in summary (should not happen)
            continue

        first_gen = q["generations"][0] if q["generations"] else ""
        prompt = _extract_prompt(first_gen)
        token_pos = _token_positions(prompt)

        record = {
            "question_id": qid,
            "prompt": prompt,
            "prob_verb_agg": summary_map[qid]["prob_verb_agg"],
            "prob_verb_match": summary_map[qid]["prob_verb_match"],
            "token_pos": token_pos,
            "original_verbalizes_hint": summary_map[qid]["original_verbalizes_hint"],
        }
        probing_records.append(record)

    # 3. Write out
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(probing_records, f, indent=2)

    # Explicitly free memory
    del raw_data
    gc.collect()

    print(f"Saved {len(probing_records)} records to {OUTPUT_JSON.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main() 