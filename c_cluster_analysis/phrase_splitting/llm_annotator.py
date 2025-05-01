from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional
import ast
import warnings

from pydantic import BaseModel
from tqdm import tqdm

# import google.generativeai as genai
from google import genai
from google.genai import types


_HAS_GENERATIVE_MODEL = hasattr(genai, "GenerativeModel")


def _gen_cfg(candidate_count: int = 1):
    base_cfg = {"temperature": 0.2, "candidate_count": candidate_count}
    if _HAS_GENERATIVE_MODEL:
        return base_cfg
    try:
        from google.generativeai import types as _old_types
        if hasattr(_old_types, "GenerateContentConfig"):
            return _old_types.GenerateContentConfig(**base_cfg)
    except Exception:
        pass
    return base_cfg

CATEGORY_DEFINITIONS = """
problem_restating: paraphrase or reformulation of the prompt to highlight givens/constraints; example words: "in other words", "the problem states", "we need to find", "I need to figure out";
knowledge_augmentation: injection of factual domain knowledge not present in the prompt; example words: "by definition", "recall that", "in general", "in cryptography, there are public and private keys";
assumption_validation: creation of examples or edge-cases to test the current hypothesis; example words: "try plugging in", "suppose", "take, for instance";
logical_deduction: logical chaining of earlier facts/definitions into a new conclusion; example words: "that would mean GDP is $15 million", "that's not matching", "Step-by-step explanation";
option_elimination: systematic ruling out of candidate answers or branches to narrow possibilities; example words: "this seems (incorrect/off)", "can’t be", "rule out";
uncertainty_or_certainty_expression: statement of confidence or doubt about the current reasoning; example words: "I'm not sure", "maybe", "I'm getting confused", "does it make sense", "Hmm, this seems a bit off", "This seems plausible", "This sounds accurate";
backtracking: abandonment of the current line of attack in favour of a new strategy; example words: "Let me think again", "on second thought", "let me rethink";
forward_planning: outline of the next intended step(s) or overall plan before executing them; example words: "first I'll", "next we should", "then I'll verify", "the plan is to", "after that we'll";
decision_confirmation: marking an intermediate result or branch as now settled; example words: "now we know", "so we've determined";
answer_reporting: presentation of the final answer with no further reasoning; example words: "final answer:", "result:";
option_restating: restating a particular MCQ option; example words: "option A was", "lets remember what option B was", "option C", "option D";
other: if none of the other labels fits, and you use 'other', then give it a short label in the other_label_field, otherwise set it to none
"""

class SegmentAnnotation(BaseModel):
    segment: str
    scores: Dict[str, float]
    other_label: Optional[str] = None

class SegmentAnnotations(BaseModel):
    annotations: List[SegmentAnnotation]


def _extract_cot_span(text: str) -> str:
    start = text.find("<think>")
    if start != -1:
        text = text[start + len("<think>"):]
    end = text.find("</think>")
    if end != -1:
        text = text[:end]
    return text.strip()


def _build_prompt(question: str, cot: str) -> str:
    return (
        "You are an expert chain-of-thought *segment* classification agent.\n"
        "Your tasks:\n"
        "1. Read the full reasoning below (between triple back-ticks).\n"
        "2. Split it into segments whenever the type of reasoning changes - even if that is within a sentence. "
        "This means, always split, if it changes from doing one of those things to rather doing another: problem restating, knowledge augmentation, logical deduction, option elimination, uncertainty or certainty expression, backtracking, forward planning, decision confirmation, answer reporting, option restating, other reasoning."
        "3. For **each** segment, assign at least one category a confidence 0-1. "
        f"{CATEGORY_DEFINITIONS}\n\n"
        "Return *only* valid JSON with this shape:\n"
        "{\n"
        '  "annotations": [\n'
        "    {\n"
        '      "segment": str,                 # the text of the segment\n'
        '      "scores": {                     # key = category name\n'
        '        "<category>": float,          # value = confidence\n'
        "        ...\n"
        "      }\n"
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        "Original multiple-choice question (context - do **not** segment this):\n"
        f"{{{question}}}\n\n"
        "```chain-of-thought to be segmented\n"
        f"{cot}\n"
        "```"
    )


def _strip_fences_and_comments(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:json)?|```?\s*$", "", txt,
                 flags=re.IGNORECASE | re.MULTILINE).strip()
    txt = re.sub(r"(?:#|//).*?$", "", txt, flags=re.MULTILINE)
    return txt

def _kill_trailing_commas(txt: str) -> str:
    return re.sub(r",\s*(\}|])", r"\1", txt)

_BAD_ESCAPE_RE = re.compile(
    r"""
    \\ 
    (?!
       ["\\/bfnrtu]
    )
    """,
    re.VERBOSE,
)

def _escape_bad_backslashes(txt: str) -> str:
    return _BAD_ESCAPE_RE.sub(r"\\\\", txt)

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

_json_kw = {"null": "None", "true": "True", "false": "False"}

def _fallback_literal_eval(txt: str):
    txt = re.sub(r'(?m)^\s*([A-Za-z_][\w]*)\s*:', r'"\1":', txt)
    txt = re.sub(r"'", r'"', txt)
    for j, p in _json_kw.items():
        txt = re.sub(rf"\b{j}\b", p, txt)
    txt = re.sub(r'([}\]0-9"]) *\n *(")', r'\1,\n\2', txt)
    return ast.literal_eval(txt)

def _parse_json(raw: str) -> Dict[str, Any]:
    txt = _escape_bad_backslashes(
        _kill_trailing_commas(
            _strip_fences_and_comments(raw)
        )
    )

    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*?\}|\[[\s\S]*?]", txt)
        if not m:
            raise
        chunk = _escape_bad_backslashes(
                    _kill_trailing_commas(m.group(0))
                )
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            # last resort
            data = _fallback_literal_eval(chunk)

    if "annotations" not in data:
        raise ValueError("Expected top-level key 'annotations'")

    cleaned: List[Dict[str, Any]] = []
    for seg in data["annotations"]:
        cleaned.append(
            {
                "segment": seg.get("segment", "").strip(),
                "scores": {k: _safe_float(v)
                           for k, v in seg.get("scores", {}).items()},
            }
        )
    return {"annotations": cleaned}

def _make_model(model_name: str, api_key: Optional[str] = None):
    if _HAS_GENERATIVE_MODEL:
        return genai.GenerativeModel(model_name)
    client = genai.Client(api_key=api_key)
    return client, model_name

def _candidate_to_text(cand) -> str:
    if cand is None:
        return ""

    if isinstance(cand, dict):
        if "text" in cand:
            return cand["text"]
        if "content" in cand and isinstance(cand["content"], dict):
            parts = cand["content"].get("parts", [])
            return "".join(p.get("text", "") if isinstance(p, dict) else str(p)
                           for p in parts)

    if hasattr(cand, "text"):
        return cand.text

    if hasattr(cand, "content"):
        cont = cand.content
        parts = getattr(cont, "parts", None)
        if parts:
            out = []
            for p in parts:
                if isinstance(p, str):
                    out.append(p)
                elif hasattr(p, "text"):
                    out.append(p.text)
                else:
                    out.append(str(p))
            return "".join(out)

    # Fallback
    return str(cand)

def _call(model_handle, prompt: str, n_samples: int = 1) -> str:
    cfg = _gen_cfg(candidate_count=n_samples)

    if _HAS_GENERATIVE_MODEL and not isinstance(model_handle, tuple):
        response = model_handle.generate_content(prompt, generation_config=cfg)

        txt = getattr(response, "text", None)
        if not txt and getattr(response, "candidates", None):
            txt = _candidate_to_text(response.candidates[0])
        return txt or ""

    client, name = model_handle

    if hasattr(client, "models") and hasattr(client.models, "generate_content"):
        rsp = client.models.generate_content(
            model=name, contents=prompt, config=cfg
        )
    else:
        rsp = client.generate_content(model=name, contents=prompt, config=cfg)

    if getattr(rsp, "candidates", None):
        return _candidate_to_text(rsp.candidates[0])
    return _candidate_to_text(rsp)


def _annotate(model_name: str,
              api_key: str | None,
              question: str,
              cot: str,
              n_samples: int) -> Dict[str, Any]:
    prompt = _build_prompt(question, cot)
    raw = _call(_make_model(model_name, api_key), prompt, n_samples)
    return _parse_json(raw)

def run_annotation_pipeline(
    completions_file: str,
    questions_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-pro",
    *,
    n_samples: int = 1,
    max_items: Optional[int] = None,
    keep_ids_file: str | None = None,
):
    # whitelist
    keep_ids: set[int] | None = None
    if keep_ids_file:
        with open(keep_ids_file, encoding="utf-8") as f:
            keep_ids = set(json.load(f))

    with open(completions_file, encoding="utf-8") as f:
        completions = json.load(f)

    if keep_ids is not None:
        completions = [c for c in completions if c["question_id"] in keep_ids]

    if max_items is not None:
        completions = completions[:max_items]

    with open(questions_file, encoding="utf-8") as f:
        q_raw = json.load(f)

    if isinstance(q_raw, dict):
        get_q = lambda qid: q_raw[str(qid)]
    else:
        q_map = {int(q["question_id"]): q["question"] for q in q_raw}
        get_q = lambda qid: q_map[qid]

    results: List[Dict[str, Any]] = []
    for entry in tqdm(completions, desc="Annotating", unit="CoT"):
        qid = entry["question_id"]
        cot_span = _extract_cot_span(entry["completion"])
        ann = _annotate(model_name, api_key, get_q(qid), cot_span, n_samples)
        results.append({"question_id": qid, "annotations": ann["annotations"]})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


__all__ = ["run_annotation_pipeline"]
