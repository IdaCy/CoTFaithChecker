from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel
from tqdm import tqdm
import ast
import warnings

# import google.generativeai as genai
from google import genai
from google.genai import types


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
uncertainty_or_certainty_expression: statement of confidence or doubt about the current reasoning; example words: "I'm not sure", "maybe", "I'm getting confused", "does it make sense", "Hmm, this seems a bit off", "This seems plausible", "This sounds accurate";
backtracking: abandonment of the current line of attack in favour of a new strategy; example words: "Let me think again", "on second thought", "let me rethink";
forward_planning: outline of the next intended step(s) or overall plan before executing them; example words: "first I'll", "next we should", "then I'll verify", "the plan is to", "after that we'll";
decision_confirmation: marking an intermediate result or branch as now settled; example words: "now we know", "so we've determined";
answer_reporting: presentation of the final answer with no further reasoning; example words: "final answer:", "result:";
option_restating: restating a particular MCQ option; example words: "option A was", "lets remember what option B was", "option C", "option D";
other: if your confidencence label of 'other' is above 0.5 then give it a short label in the other_label_field, otherwise set it to none
"""

RESPONSE_SCHEMA = """{
  "annotations": [
    {
      "sentence_id": int, 
      "problem_restating": float,
      "knowledge_augmentation": float,
      "assumption_validation": float,
      "logical_deduction": float,
      "option_elimination": float,
      "uncertainty_or_certainty_expression": float,
      "backtracking": float,
      "forward_planning": float,
      "decision_confirmation": float,
      "answer_reporting": float,
      "option_restating": float,
      "other": float,
      "other_label": str | null
    }
  ]
}
"""


CATEGORY_NAMES: List[str] = [
    "problem_restating",
    "knowledge_augmentation",
    "assumption_validation",
    "logical_deduction",
    "option_elimination",
    "uncertainty_or_certainty_expression",
    "backtracking",
    "forward_planning",
    "decision_confirmation",
    "answer_reporting",
    "option_restating",
    "other",
]

#CategoryVec = conlist(float, min_items=12, max_items=12)

class Annotation_1(BaseModel):
    sentence_id: int
    categories: List[float]
    other_label: Optional[str] = None

class Annotations_1(BaseModel):
    annotations: List[Annotation_1]

class Annotation_2(BaseModel):
    sentence_id: int
    problem_restating: float
    knowledge_augmentation: float
    assumption_validation: float
    logical_deduction: float
    option_elimination: float
    uncertainty_or_certainty_expression: float
    backtracking: float
    forward_planning: float
    decision_confirmation: float
    answer_reporting: float
    option_restating: float
    other: float
    other_label: Optional[str] = None

class Annotations_2(BaseModel):
    annotations: List[Annotation_2]

CoTAnnotation = Annotations_1

def _truncate_or_pad(seq, target):
    if len(seq) >= target:
        return seq[:target]
    return seq + [0.0] * (target - len(seq))

def convert_annotations_1_to_2(annotations_1: Annotations_1) -> Annotations_2:
    """
    Converts an Annotations_1 object to an Annotations_2 object.

    Assumes the order of floats in Annotation_1.categories corresponds directly
    to the order of the float fields in Annotation_2.
    """
    # Define the field names in the exact order they appear in Annotation_2
    category_field_names = [
        "problem_restating",
        "knowledge_augmentation",
        "assumption_validation",
        "logical_deduction",
        "option_elimination",
        "uncertainty_or_certainty_expression",
        "backtracking",
        "forward_planning",
        "decision_confirmation",
        "answer_reporting",
        "option_restating",
        "other",
    ]

    NUM = len(category_field_names)
    new_annotations: List[Annotation_2] = []

    for ann_1 in annotations_1.annotations:
        if len(ann_1.categories) != NUM:
            warnings.warn(
                f"sentence_id {ann_1.sentence_id}: got {len(ann_1.categories)} "
                f"values, expected {NUM} - trimming/padding", RuntimeWarning
            )

        """cats = _truncate_or_pad(ann_1.categories, NUM)
        category_data = {field: cats[i] for i, field in enumerate(category_field_names)}

        for ann_1 in annotations_1.annotations:
        if len(ann_1.categories) != len(category_field_names):
            raise ValueError(
                f"Annotation_1 with sentence_id {ann_1.sentence_id} has "
                f"{len(ann_1.categories)} categories, but Annotation_2 expects "
                f"{len(category_field_names)}."
            )"""

        cats = _truncate_or_pad(ann_1.categories, NUM)
        category_data = dict(zip(category_field_names, cats))

        new_annotations.append(
            Annotation_2(
                sentence_id=ann_1.sentence_id,
                other_label=ann_1.other_label,
                **category_data,
            )
        )

        """# Create a dictionary mapping field names to category values
        category_data = {
            field_name: ann_1.categories[i]
            for i, field_name in enumerate(category_field_names)
        }

        # Create the Annotation_2 object
        ann_2 = Annotation_2(
            sentence_id=ann_1.sentence_id,
            other_label=ann_1.other_label,
            **category_data  # Unpack the category data
        )
        new_annotations.append(ann_2)"""

    return Annotations_2(annotations=new_annotations)

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
    start = text.find("<think>")
    if start != -1:
        text = text[start + len("<think>"):]
    end = text.find("</think>")
    if end != -1:
        text = text[:end]

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
        "You are an expert chain-of-thought classification agent.\n"
        "For **each** sentence below, assign a confidence score (0-1) for **every** "
        f"category, according to how well it matches with the description. Note that more than one cateogory is allowed to have non-zero scores, but one category has to have the highest score:\n\n"
        f"{CATEGORY_DEFINITIONS}\n\n"
        "Special rule for *other*: if the score you give to *other* is > 0.5, also add "
        "`other_label` with a very short description (1-5 words). Otherwise set "
        "`other_label` to None.\n\n"
        "For context - the original input question (do **NOT** label):\n"
        f"{{{question}}}\n\n"
        "Sentences:\n"
        f"{numbered}\n\n"
        "Return **only** JSON that matches the response_schema."
        "Please return the confidence for each category in the list in this EXACT order: [problem_restating, knowledge_augmentation, assumption_validation, logical_deduction, option_elimination, uncertainty_or_certainty_expression, backtracking, forward_planning, decision_confirmation, answer_reporting, option_restating, other]."
        "The list **must exactly contain 12 numbers** in the order above."
    )

def _make_model(model_name: str, api_key: Optional[str] = None):
    if _HAS_GENERATIVE_MODEL:
        return genai.GenerativeModel(model_name)
    client = genai.Client(api_key=api_key)
    return client, model_name


def _call(model_handle, prompt: str, n_samples: int = 1) -> str:
    cfg = _gen_cfg(candidate_count=n_samples)

    if _HAS_GENERATIVE_MODEL:
        response = model_handle.generate_content(prompt, generation_config=cfg)
        raw = response.text
    else:
        client, name = model_handle
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            rsp = client.models.generate_content(model=name, contents=prompt, config=cfg)
        else:
            rsp = client.generate_content(model=name, contents=prompt, config=cfg)
        raw = rsp.candidates[0].text

    return raw or ""

def _strip_fences_and_comments(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:json)?|```?\s*$", "", txt,
                 flags=re.IGNORECASE | re.MULTILINE).strip()
    txt = re.sub(r"(?:#|//).*?$", "", txt, flags=re.MULTILINE)
    return txt


def _kill_trailing_commas(txt: str) -> str:
    return re.sub(r",\s*(\}|])", r"\1", txt)


_json_kw = {"null": "None", "true": "True", "false": "False"}

def _fallback_literal_eval(txt: str):
    txt = re.sub(r'(?m)^\s*([A-Za-z_][\w]*)\s*:', r'"\1":', txt)
    txt = re.sub(r"'", r'"', txt)
    for j, p in _json_kw.items():
        txt = re.sub(rf"\b{j}\b", p, txt)
    txt = re.sub(r'([}\]0-9"]) *\n *(")', r'\1,\n\2', txt)
    return ast.literal_eval(txt)

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
        if m:
            chunk = _escape_bad_backslashes(
                        _kill_trailing_commas(m.group(0))
                    )
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                data = _fallback_literal_eval(chunk)
        else:
            raise

    if isinstance(data, list):
        data = {"annotations": data}
    elif isinstance(data, dict) and "annotations" not in data:
        for v in data.values():
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                data = {"annotations": v}
                break
    if "annotations" not in data:
        raise ValueError("Could not find an 'annotations' list in model output")

    cleaned: List[Dict[str, Any]] = []
    for idx, item in enumerate(data["annotations"], 1):
        rec = {"sentence_id": int(item.get("sentence_id", idx))}

        scores: Dict[str, Any] = {k: item[k] for k in CATEGORY_NAMES if k in item}

        if not scores:
            for key in ("scores", "confidences", "confidence_scores"):
                if key in item and isinstance(item[key], dict):
                    scores = item[key]
                    break

        if not scores:
            for key in ("confidences", "scores", "values", "probabilities"):
                if key in item and isinstance(item[key], list):
                    lst = item[key]
                    if len(lst) >= len(CATEGORY_NAMES):
                        scores = dict(zip(CATEGORY_NAMES, lst))
                    break
            else:
                if isinstance(item, list) and len(item) >= len(CATEGORY_NAMES):
                    scores = dict(zip(CATEGORY_NAMES, item))

        for cat in CATEGORY_NAMES:
            rec[cat] = float(scores.get(cat, 0.0))
        """total = sum(rec[c] for c in CATEGORY_NAMES)
        if total > 0:
            for c in CATEGORY_NAMES:
                rec[c] = rec[c] / total"""
        rec["other_label"] = item.get("other_label")
        cleaned.append(rec)

    return {"annotations": cleaned}


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
        if sid is None:                                 # fallback: parse [12]
            sent_txt = item.get("sentence") or item.get("text", "")
            m_id = re.search(r"\[(\\d+)]", sent_txt)
            sid = int(m_id.group(1)) if m_id else idx
        cat = (item.get("categories") or item.get("category")
               or item.get("label") or item.get("annotation"))
        if cat is None:
            continue

        norm.append({"sentence_id": sid, "categories": cat})

    return {"annotations": norm}



def call_gemini(model_name, api_key, prompt: str):

    client = genai.Client(api_key=api_key)

    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=500,
        ),
        response_mime_type="application/json",
        response_schema=Annotations_1,
    )

    response = client.models.generate_content(
        model=model_name,
        contents= prompt,
        config=generate_content_config,
    )

    return response.parsed

def _annotate(model_name, api_key, question: str, sentences: List[str], n_samples: int) -> Annotations_2:
    prompt = _build_prompt(question, sentences)

    #print("\n" + "‾" * 80 + "\nPROMPT SENT TO GEMINI:\n" + prompt + "\n" + "_" * 80)

    # raw = _call(model_handle, prompt, n_samples=n_samples)
    # #print("\nRAW GEMINI RESPONSE:\n", raw, "\n" + "-"*80)
    # data = _parse_json(raw)
    #eturn Annotations(**data)
    #annotations_2 = annotations_to_annotations_2(annotations)

    preliminary_annotation = call_gemini(model_name, api_key, prompt)
    final_annotations = convert_annotations_1_to_2(preliminary_annotation)

    return final_annotations

# def _annotate(model_handle, question: str, sentences: List[str], n_samples: int) -> Annotations_1:
#     prompt = _build_prompt(question, sentences)

#     #print("\n" + "‾" * 80 + "\nPROMPT SENT TO GEMINI:\n" + prompt + "\n" + "_" * 80)

#     raw = _call(model_handle, prompt, n_samples=n_samples)
#     #print("\nRAW GEMINI RESPONSE:\n", raw, "\n" + "-"*80)
#     data = _parse_json(raw)
#     #eturn Annotations(**data)
#     #annotations_2 = annotations_to_annotations_2(annotations)
    
#     preliminary_annotation = Annotations_1(**data)
#     final_annotations = annotations_to_annotations_2(preliminary_annotation)

#     return final_annotations

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
    # ID whitelist
    keep_ids: set[int] | None = None
    if keep_ids_file:
        with open(keep_ids_file, encoding="utf-8") as f:
            keep_ids = set(json.load(f))
            print("keep_ids")
            print(keep_ids)

    """# whitelist filter
    with open(completions_file, encoding="utf-8") as f:
        completions = json.load(f)
        if keep_ids is not None:
            completions = [c for c in completions if c["question_id"] in keep_ids]
        if max_items is not None:
            completions = completions[:max_items]

    # key = _configure_genai(api_key)
    # model_handle = _make_model(model_name, key) if not _HAS_GENERATIVE_MODEL else _make_model(model_name)

    with open(completions_file, encoding="utf-8") as f:
        completions = json.load(f)
        if max_items is not None:
            completions = completions[:max_items]"""

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
        sentences = _split_sentences(entry["completion"])
        ann = _annotate(model_name, api_key, get_q(qid), sentences, n_samples)
        results.append({"question_id": qid, "annotations": [a.dict() for a in ann.annotations]})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

__all__ = ["run_annotation_pipeline"]
