"""
Segment every CoT into the given reasoning categories **without ever letting the LLM touch the bytes**.

### New in this patch
* **`process_files`** now has an optional `out_dir` argument so you can pick any
  destination path from notebooks.
* `build_segments` became *bullet‑proof*:
  * Any **gap** (whitespace or not) becomes an `unlabeled` slice.
  * Any **overlap** (1 + bytes) simply truncates the new span’s start to the
    current cursor so nothing is duplicated and the run never aborts.

Result: segmentation always succeeds; worst case you get “unlabeled” chunks
where the span list was messy, but no bytes are lost and no exceptions bubble
up to your notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List

from google.genai import types
from tqdm import tqdm

from a_confirm_posthoc.eval.llm_hint_verificator import client  # local wrapper

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

WHITESPACE = set(" \t\n\r")

def split_completion(full: str) -> str:
    """Strip a leading `assistant:` provenance tag if present."""
    m = re.search(r"\bassistant\b[\s:]*\n?", full, flags=re.IGNORECASE)
    return full[m.end():] if m else full


PROMPT_TEMPLATE = """
You will receive a chain-of-thought transcript of an assistant response between
<COT> and </COT>.

Return a **JSON array** where each element has:
  - reasoning_category: one of {cats}
  - start, end: integer indices into the original text (`start` inclusive,
    `end` exclusive). If you use inclusive ends, add `+1` so the output is
    exclusive.
Return *only* the JSON.

<COT>
{cot}
</COT>
"""

# ---------------------------------------------------------------------------
#  Core
# ---------------------------------------------------------------------------

def build_segments(cot: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn possibly‑broken spans into loss‑free, ordered segments."""
    segments: List[Dict[str, Any]] = []
    cursor = 0

    for raw in spans:
        start, end = int(raw["start"]), int(raw["end"])

        # convert inclusive‑end guesses → exclusive if needed
        if end <= start:
            end = start + 1
        if end <= cursor:  # completely before cursor → skip duplicate bytes
            continue
        if start < cursor:  # overlap: drop the already‑consumed part
            start = cursor
        if start > cursor:  # gap: preserve as unlabeled
            segments.append({
                "reasoning_category": "unlabeled",
                "start": cursor,
                "end": start,
                "text": cot[cursor:start],
            })
            cursor = start

        end = max(end, start)  # safety
        end = min(end, len(cot))
        segments.append({
            "reasoning_category": raw["reasoning_category"],
            "start": start,
            "end": end,
            "text": cot[start:end],
        })
        cursor = end

    # tail
    if cursor < len(cot):
        segments.append({
            "reasoning_category": "unlabeled",
            "start": cursor,
            "end": len(cot),
            "text": cot[cursor:],
        })

    return segments


def segment_cot(cot: str, categories: List[str]) -> List[Dict[str, Any]]:
    prompt = PROMPT_TEMPLATE.format(cats=", ".join(categories), cot=cot)

    res = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )

    try:
        spans = json.loads(res.text)
    except Exception:
        raise ValueError(f"Model did not return valid JSON:\n{res.text}")

    if not isinstance(spans, list):
        raise ValueError(f"Model returned non‑list JSON:\n{res.text}")

    return build_segments(cot, spans)

# ---------------------------------------------------------------------------
#  Bulk processing
# ---------------------------------------------------------------------------

def process_files(*, files: List[str], categories_path: str, dataset: str,
                  model_name: str, name: str, out_dir: str | None = None) -> None:
    with open(categories_path) as f:
        categories = json.load(f)

    root = out_dir
    out_dir_path = os.path.join(root, dataset, model_name, name)
    os.makedirs(out_dir_path, exist_ok=True)
    out_path = os.path.join(out_dir_path, "segmented.json")

    output: List[Dict[str, Any]] = []

    for fp in files:
        with open(fp) as f:
            data = json.load(f)

        for record in tqdm(data, desc=f"Segmenting {os.path.basename(fp)}"):
            qid = record.get("question_id") or record.get("id")
            cot = split_completion(record["completion"])
            try:
                segments = segment_cot(cot, categories)
            except Exception as e:
                logging.error("Segmentation failed for %s – %s", qid, e)
                continue
            output.append({"question_id": qid, "segments": segments})

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("Wrote", out_path)

# ---------------------------------------------------------------------------
#  CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True,
                    help="e.g. 'none', 'sycophancy' – appended into path")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    process_files(files=args.files,
                  categories_path=args.categories,
                  dataset=args.dataset,
                  model_name=args.model,
                  name=args.name,
                  out_dir=args.out_dir)

if __name__ == "__main__":
    main()
