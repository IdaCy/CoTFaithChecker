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
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME = MODEL_PATH.split("/")[-1]
HINT_TYPE = "sycophancy"         # e.g. "sycophancy"
N_QUESTIONS = 5001     # e.g. 301

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

# We rely on the model tokenizer to obtain accurate token boundaries. Some model names
# (e.g. DeepSeek-R1-Distill-Llama-8B) may not be available on the Hugging Face hub or
# might require custom loading logic. We therefore attempt to load the tokenizer but
# gracefully fall back to whitespace tokenisation if that fails.

THINK_DELIM = "<think>\n"  # delimiter that separates prompt from assistant reasoning


try:
    from transformers import AutoTokenizer  # type: ignore

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
except Exception as _e:  # pragma: no cover – informative warning only
    print(
        f"[create_probing_dataset] Warning: could not load tokenizer for '{MODEL_NAME}'. "
        "Falling back to naive whitespace tokenisation – token indices may be inaccurate."
    )
    _TOKENIZER = None


def _compute_probs(agg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Compute prob_verb_match from aggregated_counts."""
    match_hint = agg["match_hint_count"]

    prob_verb_match = (
        agg["match_and_verbalize_count"] / match_hint if match_hint else None
    )
    return {
        "prob_verb_match": prob_verb_match,
    }


def _extract_prompt(gen_str: str) -> str:
    """Return the prompt portion up to and including the last `<think>\n`."""
    idx = gen_str.rfind(THINK_DELIM)
    if idx == -1:
        # Fallback: return entire string if delimiter missing
        return gen_str
    return gen_str[: idx + len(THINK_DELIM)]


def _find_hint_token_idx(prompt: str) -> int:
    """Return the *token index* (0-based) corresponding to the hinted answer in the prompt.

    The heuristic is: locate the last occurrence of '[' (opening bracket) in the raw
    prompt and take the very next *token* that begins at (or immediately after) that
    position. This works for prompts of the form "... pointing to [ C ]." where the
    bracketed letter is the hinted answer. If the bracket cannot be found or the
    tokenizer fails, returns -1.
    """

    # Locate the last '[' character – this precedes the hinted option (e.g. "[ C ]")
    char_idx = prompt.rfind("[")
    if char_idx == -1:
        raise ValueError("No hint token found in prompt")

    if _TOKENIZER is None:
        # Naive split fallback
        raise ValueError("No tokenizer found")

    # Use the tokenizer's encode method (without special tokens) to count tokens before
    # the '[' character. Offsets are not returned by slow tokenisers, so we simply
    # re-encode the substring up to the bracket.
    sub_tokens = _TOKENIZER.encode(prompt[:char_idx], add_special_tokens=False)
    return len(sub_tokens)


def _token_positions(prompt: str) -> List[int]:
    """Return `[assistant_idx, think_idx, hint_idx]` for the given prompt.

    • `assistant_idx` – token index of the ``<|Assistant|>`` header immediately before
      the final `<think>` delimiter. This is always the **third** token from the end
      of the prompt, so we record it as `-3` (relative index).

    • `think_idx` – token index of the `<think>` delimiter, which is always the second
      token from the end (`-2`).

    • `hint_idx` – token index (0-based, counting from the *start* of the prompt) of
      the hinted answer option (e.g. the `'C'` in "[ C ]"). If the hint cannot be
      located the function returns `-1` for this element.
    """

    # Assistant and <think> positions are fixed relative to the end of the prompt
    assistant_idx_rel = -3  # third token from the end
    think_idx_rel = -2      # second token from the end

    hint_idx_abs = _find_hint_token_idx(prompt)

    return [assistant_idx_rel, think_idx_rel, hint_idx_abs]


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
            "prob_verb_match": summary_map[qid]["prob_verb_match"],
            "token_pos": token_pos,
            "original_verbalizes_hint": summary_map[qid]["original_verbalizes_hint"],
        }
        probing_records.append(record)

        # Sanity check: Print tokens at calculated positions
        # if _TOKENIZER:
        #     try:
        #         tokens = _TOKENIZER.tokenize(prompt)
        #         asst_idx_abs = len(tokens) + token_pos[0]  # Convert relative -3 to absolute
        #         think_idx_abs = len(tokens) + token_pos[1] # Convert relative -2 to absolute
        #         hint_idx_abs = token_pos[2]

        #         print(f"--- QID: {qid} ---")
        #         print(f"Token @ Assistant Idx ({token_pos[0]} -> {asst_idx_abs}): '{tokens[asst_idx_abs]}'")
        #         print(f"Token @ Think Idx ({token_pos[1]} -> {think_idx_abs}): '{tokens[think_idx_abs]}'")
        #         if hint_idx_abs != -1:
        #             print(f"Token @ Hint Idx ({hint_idx_abs}): '{tokens[hint_idx_abs]}'")
        #         else:
        #             print("Hint token not found.")
        #         print("-" * (13 + len(str(qid)))) # Match width of header line
        #     except IndexError:
        #         print(f"[create_probing_dataset] Warning: QID {qid} - Token index out of range during sanity check.")
        #     except Exception as e:
        #         print(f"[create_probing_dataset] Warning: QID {qid} - Error during sanity check: {e}")

    # 3. Write out
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(probing_records, f, indent=2)

    # Explicitly free memory
    del raw_data
    gc.collect()

    print(f"Saved {len(probing_records)} records to {OUTPUT_JSON}")


if __name__ == "__main__":
    main() 