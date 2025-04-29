from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel
from tqdm import tqdm

import google.generativeai as genai

_HAS_GENERATIVE_MODEL = hasattr(genai, "GenerativeModel")
def _gen_cfg(candidate_count: int = 1):
    """Return an SDK‑compatible generation_config structure."""
    base_cfg = {"temperature": 0.2, "candidate_count": candidate_count}

    if _HAS_GENERATIVE_MODEL:
        return base_cfg
    try:
        from google.generativeai import types as _old_types
        if hasattr(_old_types, "GenerateContentConfig"):
            cfg = _old_types.GenerateContentConfig(**base_cfg)
            return cfg
    except Exception:
        pass

    return base_cfg

CATEGORY_DEFINITIONS = """
problem_restating: paraphrase or reformulation of the prompt to highlight givens/constraints; example words: "in other words", "the problem states", "we need to find", "I need to figure out";
knowledge_augmentation: injection of factual domain knowledge not present in the prompt; example words: "by definition", "recall that", "in general", "in cryptography, there are public and private keys";
assumption_validation: creation of examples or edge-cases to test the current hypothesis; example words: "try plugging in", "suppose", "take, for instance";
logical_deduction: logical chaining of earlier facts/definitions into a new conclusion; example words: "that would mean GDP is $15 million", "that's not matching", "Step-by-step explanation";
option_elimination: systematic ruling out of candidate answers or branches to narrow possibilities; example words: "this seems (incorrect/off)", "can’t be", "rule out";
uncertainty_expression: statement of confidence or doubt about the current reasoning; example words: "I'm not sure", "maybe", "I'm getting confused", "does it make sense", "Hmm, this seems a bit off";
backtracking: abandonment of the current line of attack in favour of a new strategy; example words: "Let me think again", "on second thought", "let me rethink";
decision_confirmation: marking an intermediate result or branch as now settled; example words: "now we know", "so we've determined";
answer_reporting: presentation of the final answer with no further reasoning; example words: "final answer:", "result:"
"""

class Annotation(BaseModel):
    sentence_id: int
    categories: str

class Annotations(BaseModel):
    annotations: List[Annotation]

CoTAnnotation = Annotations

def _configure_genai(api_key: str | None) -> str:
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("Gemini API key missing (arg or env var)")
    genai.configure(api_key=key)
    return key

"""def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s for s in re.split(r"(?<=\.)\s+", text) if s]"""

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    parts = [p.strip() for p in re.split(r"(?<=\.)\s+", text) if p.strip()]

    merged: List[str] = []
    i = 0
    while i < len(parts):
        if re.fullmatch(r"\d+\.", parts[i]) and i + 1 < len(parts):
            merged.append(f"{parts[i]} {parts[i + 1]}")
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return merged

def _build_prompt(question: str, sentences: Sequence[str]) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
    return (
        "You are an expert chain‑of‑thought classification agent. Assign **exactly one** "
        "category from the list below to **each** sentence.\n\n"
        f"{CATEGORY_DEFINITIONS}\n\n"
        "Input question (context, do **NOT** label):\n"
        f"{{{question}}}\n\n"
        "Sentences:\n"
        f"{numbered}\n\n"
        "Return **only** JSON that matches the response_schema."
    )

def _make_model(model_name: str, api_key: Optional[str] = None):
    if _HAS_GENERATIVE_MODEL:
        return genai.GenerativeModel(model_name)
    # Legacy path – need client + model name
    client = genai.Client(api_key=api_key)
    return client, model_name


def _call(model_handle, prompt: str, n_samples: int = 1) -> str:
    cfg = _gen_cfg(candidate_count=n_samples)

    if _HAS_GENERATIVE_MODEL:
        response = model_handle.generate_content(prompt, generation_config=cfg)
        raw = response.text  # first candidate’s text
    else:
        client, name = model_handle
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            rsp = client.models.generate_content(model=name, contents=prompt, config=cfg)
        else:
            rsp = client.generate_content(model=name, contents=prompt, config=cfg)
        raw = rsp.candidates[0].text

    return raw or ""

def _parse_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw,
                 flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()

    def _loads(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    obj = _loads(raw)
    if obj is None:
        #m = re.search(r"\\{.*\\}|\\[.*\\]", raw, flags=re.DOTALL)
        m = re.search(r"\{.*\}|\[.*\]", raw, flags=re.DOTALL)
        if m:
            obj = _loads(m.group(0))
    if obj is None:
        raise json.JSONDecodeError("Could not parse JSON", raw, 0)

    if isinstance(obj, dict) \
       and all(re.fullmatch(r"\d+", k) for k in obj.keys()) \
       and all(isinstance(v, str) for v in obj.values()):
        return {
            "annotations": [
                {"sentence_id": int(k), "categories": v}
                for k, v in obj.items()
            ]
        }

    def _find_list(o):
        if isinstance(o, list) and all(isinstance(x, dict) for x in o):
            return o
        if isinstance(o, dict):
            if o and all(isinstance(v, dict) for v in o.values()):
                return list(o.values())
            for v in o.values():
                hit = _find_list(v)
                if hit is not None:
                    return hit
        return None

    annotations = _find_list(obj)
    if annotations is None:
        raise ValueError("Could not locate a list of sentence annotations in Gemini response")

    norm: List[Dict[str, Any]] = []
    for idx, item in enumerate(annotations, 1):
        if not isinstance(item, dict):
            continue

        sid = item.get("sentence_id")
        if sid is None:                                 # fallback: parse “[12] …”
            sent_txt = item.get("sentence") or item.get("text", "")
            m_id = re.search(r"\[(\\d+)]", sent_txt)
            sid = int(m_id.group(1)) if m_id else idx
        cat = (item.get("categories") or item.get("category")
               or item.get("label") or item.get("annotation"))
        if cat is None:
            continue

        norm.append({"sentence_id": sid, "categories": cat})

    return {"annotations": norm}

def _annotate(model_handle, question: str, sentences: List[str], n_samples: int) -> Annotations:
    prompt = _build_prompt(question, sentences)

    print("\n" + "‾" * 80 + "\nPROMPT SENT TO GEMINI:\n" + prompt + "\n" + "_" * 80)

    raw = _call(model_handle, prompt, n_samples=n_samples)
    #print("\nRAW GEMINI RESPONSE:\n", raw, "\n" + "-"*80)
    data = _parse_json(raw)
    return Annotations(**data)

def run_annotation_pipeline(
    completions_file: str,
    questions_file: str,
    output_file: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-pro",
    *,
    n_samples: int = 1,
    max_items: Optional[int] = None,
):
    key = _configure_genai(api_key)
    model_handle = _make_model(model_name, key) if not _HAS_GENERATIVE_MODEL else _make_model(model_name)

    with open(completions_file, encoding="utf-8") as f:
        completions = json.load(f)
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
        sentences = _split_sentences(entry["completion"])
        ann = _annotate(model_handle, get_q(qid), sentences, n_samples)
        results.append({"question_id": qid, "annotations": [a.dict() for a in ann.annotations]})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

__all__ = ["run_annotation_pipeline"]
