"""
A one-shot “no-CoT” evaluation: we ask the model for the answer
letter only, *without* any chain-of-thought cue.  Accuracy should
hover around chance (≈25 % for 4-way MCQ).  If it is much higher,
the test set is probably solvable without reasoning instructions
and is therefore unsuitable for a faithfulness experiment.
"""

import json, os, re, time, logging
from typing import List, Dict, Optional

import torch                   # only to satisfy type hints; not used directly

from a_confirm_posthoc.src.main.xxpipeline import get_chat_template
from a_confirm_posthoc.src.utils.model_handler import generate_completion

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

_ANSWER_RE = re.compile(r"\b([ABCD])\b", re.I)


def _build_direct_prompt(entry: Dict) -> str:
    """
    Bare-bones prompt with *no* chain-of-thought instruction.
    Always ends with an explicit request for the letter only.
    """
    return (
        f"Question: {entry['question']}\n"
        f"A. {entry['A']}\n"
        f"B. {entry['B']}\n"
        f"C. {entry['C']}\n"
        f"D. {entry['D']}\n\n"
        "Give only the letter (A, B, C or D) of the correct answer."
    )


def _first_letter(text: str) -> Optional[str]:
    """Extract the first standalone A/B/C/D from `text` (case-insensitive)."""
    m = _ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def sanity_check(
    *,
    model,
    tokenizer,
    model_name: str,
    device: torch.device,
    dataset_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 4,
    n_questions: Optional[int] = None,
    save_dir: str = "data",
) -> Dict:
    """
    Run the sanity check and (optionally) save a JSON report.

    Returns
    -------
    dict  with keys  {'accuracy', 'correct', 'total', 'path'}
    """
    t0 = time.time()
    logging.info("Running no-CoT sanity check on %s …", dataset_name)

    # ------------------------------------------------------------------ load
    data_path = os.path.join("data", dataset_name, "input_mcq_data.json")
    with open(data_path, "r") as fh:
        data: List[Dict] = json.load(fh)
    if n_questions:
        data = data[: n_questions]

    # --------------------------------------------------------------- prompts
    prompts = [
        {"question_id": e["question_id"], "prompt_text": _build_direct_prompt(e)}
        for e in data
    ]
    chat_template = get_chat_template(model_name)

    # ------------------------------------------------------- model inference
    completions = generate_completion(
        model,
        tokenizer,
        device,
        prompts,
        chat_template,
        batch_size,
        max_new_tokens,
    )

    # ------------------------------------------------------- scoring & logs
    qid2gold = {e["question_id"]: e["correct"] for e in data}
    correct = 0
    detailed: List[Dict] = []
    for c in completions:
        qid = c["question_id"]
        pred_letter = _first_letter(c["completion"])
        gold_letter = qid2gold[qid]
        is_ok = pred_letter == gold_letter
        correct += int(is_ok)
        detailed.append(
            {
                "question_id": qid,
                "prediction": pred_letter,
                "gold": gold_letter,
                "is_correct": is_ok,
                "raw_completion": c["completion"],
            }
        )

    total = len(data)
    acc = correct / total if total else 0.0
    runtime = time.time() - t0

    # -------------------------------------------------------------- persist
    out_path = os.path.join(
        save_dir,
        dataset_name,
        model_name,
        "sanity_check",
        f"sanity_check_{total}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "runtime_s": runtime,
                "results": detailed,
            },
            fh,
            indent=2,
        )

    logging.info(
        "Sanity check finished: %.2f %% (%d/%d) | saved → %s",
        100 * acc,
        correct,
        total,
        out_path,
    )

    return {"accuracy": acc, "correct": correct, "total": total, "path": out_path}
