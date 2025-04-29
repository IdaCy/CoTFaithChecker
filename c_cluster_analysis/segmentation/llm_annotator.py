# c_cluster_analysis/segmentation/llm_annotator.py
"""Version‑tolerant Gemini annotation helper.
Runs against **any** released version of `google‑generativeai` (0.1 → current)
without import errors.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel
from tqdm import tqdm

import google.generativeai as genai

_HAS_GENERATIVE_MODEL = hasattr(genai, "GenerativeModel")  # ≥ 0.3

if _HAS_GENERATIVE_MODEL:
    def _gen_cfg() -> dict:  # noqa: D401
        return {"temperature": 0.2}
else:
    try:
        from google.generativeai import types as _old_types

        def _gen_cfg() -> Any:+
            if hasattr(_old_types, "GenerateContentConfig"):
                return _old_types.GenerateContentConfig(temperature=0.2)
            return {"temperature": 0.2}

    except Exception:
        def _gen_cfg() -> dict:
            return {"temperature": 0.2}

CATEGORY_DEFINITIONS = """problem_restating: paraphrase or reformulation …
knowledge_augmentation: injection of factual knowledge …
assumption_validation: creation of examples or edge‑cases …
logical_deduction: logical chaining of earlier facts …
option_elimination: systematic ruling out of candidate answers …
uncertainty_expression: statement of confidence or doubt …
backtracking: abandoning the current line of attack …
decision_confirmation: marking an intermediate result …
answer_reporting: presentation of the final answer only."""


class Annotation(BaseModel):
    sentence_id: int
    categories: str


class Annotations(BaseModel):
    annotations: List[Annotation]


def _configure_genai(api_key: str | None) -> str:
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("Gemini API key missing (arg or env var)")
    genai.configure(api_key=key)
    return key


def _split_sentences(text: str) -> List[str]:
    """Naïve sentence splitter on period+space."""
    text = re.sub(r"\s+", " ", text.strip())
    return [s for s in re.split(r"(?<=\.)\s+", text) if s]


def _build_prompt(question: str, sentences: Sequence[str]) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
    return f"""
You are an expert chain‑of‑thought classification agent. Assign **exactly one**
category from the list below to **each** sentence.

{CATEGORY_DEFINITIONS}

Input question (context, do **NOT** label):
{{{question}}}

Sentences:
{numbered}

Return **only** JSON that matches the response_schema.
""".strip()

if _HAS_GENERATIVE_MODEL:

    def _make_model(model_name: str):  # noqa: D401
        return genai.GenerativeModel(model_name)

    def _call(model, prompt: str) -> str:  # noqa: D401
        return model.generate_content(prompt, generation_config=_gen_cfg()).text

else:
    def _make_model(model_name: str, api_key: str):  # noqa: D401
        return genai.Client(api_key=api_key), model_name

    def _call(client_and_name, prompt: str) -> str:  # noqa: D401
        client, name = client_and_name
        cfg = _gen_cfg()
        # Older sdk had both client.models.generate_content and client.generate_content.
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            rsp = client.models.generate_content(model=name, contents=prompt, config=cfg)
        else:
            rsp = client.generate_content(model=name, contents=prompt, config=cfg)  # type: ignore
        return rsp.candidates[0].text  # type: ignore

def _annotate(model_handle, question: str, sentences: List[str]) -> Annotations:
    raw = _call(model_handle, _build_prompt(question, sentences))
    return Annotations(**json.loads(raw))

def run_annotation_pipeline(
    completions_file: str,
    questions_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-pro",
):
    key = _configure_genai(api_key)
    model_handle = _make_model(model_name) if _HAS_GENERATIVE_MODEL else _make_model(model_name, key)

    with open(completions_file, encoding="utf-8") as f:
        completions = json.load(f)
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
        sentences = _split_sentences(entry["completion"])
        ann = _annotate(model_handle, get_q(qid), sentences)
        results.append({"question_id": qid, "annotations": [a.dict() for a in ann.annotations]})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

__all__ = ["run_annotation_pipeline"]
