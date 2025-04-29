from __future__ import annotations
"""cot_sentence_categorizer.py

Utilities for sentence-level categorisation of LLM Chain-of-Thought (CoT) into the
specified phrase categories.  The public entry-point is `run_cot_annotation`, which
mirrors the API of the previous `run_hint_verification` helper so that existing
notebooks can simply do::

    from cot_sentence_categorizer import run_cot_annotation

and run the pipeline on an existing directory structure.

The module expects the same folder hierarchy that was produced by the earlier
hint-verification script, i.e.::

    data/<dataset_name>/<model_name>/<hint_type>/completions_with_<N>.json

where each *completion* structure is a dict that contains at least:

    { "question_id": int, "completion": str }

`run_cot_annotation` will create, in the same folder, a file named

    cot_annotations_with_<N>.json

containing a list of objects in the required format::

    [
      {
        "question_id": 0,
        "sentence_annotations": {
          "1": {"sentence": "...", "categories": ["problem_restating"]},
          "2": {"sentence": "...", "categories": ["logical_deduction"]},
          ...
        }
      },
      ...
    ]

The categorisation itself is delegated to Gemini via the **Google Generative AI
Python SDK**, with a thinking budget of **600**.  A robust pydantic schema is
used for response validation.
"""

from typing import Dict, List
from pathlib import Path
import json
import logging
import os
import re

from pydantic import BaseModel, Field, computed_field
from google import genai
from google.genai import types

###########################################################################
# CONSTANTS & CATEGORY DEFINITIONS                                         #
###########################################################################

CATEGORY_DEFINITIONS = (
    "problem_restating: paraphrase or reformulation of the prompt to highlight givens/constraints; example words: \"in other words\", \"the problem states\", \"we need to find\", \"I need to figure out\";\n"
    "knowledge_augmentation: injection of factual domain knowledge not present in the prompt; example words: \"by definition\", \"recall that\", \"in general\", “in cryptography, there are public and private keys”;&#x5c;n"
    "assumption_validation: creation of examples or edge-cases to test the current hypothesis; example words: \"try plugging in\", \"suppose\", \"take, for instance\";&#x5c;n"
    "logical_deduction: logical chaining of earlier facts/definitions into a new conclusion; example words: “that would mean GDP is $15 million”, “that's not matching.", \"Step-by-step explanation\";&#x5c;n"
    "option_elimination: systematic ruling out of candidate answers or branches to narrow possibilities; example words: \"this seems (incorrect/off)\", \"can’t be\", \"rule out”;&#x5c;n"
    "uncertainty_expression: statement of confidence or doubt about the current reasoning; \"i’m not sure\",  \"maybe\", “I’m getting confused”, \"does it make sense\", \"Hmm, this seems a bit off\";&#x5c;n"
    "backtracking: abandonment of the current line of attack in favour of a new strategy, or consideration of another strategy (but distinct from uncertainty_expression through focus on alternative); example words: \"Let me think again”, \"on second thought\", \"let me rethink”;&#x5c;n"
    "decision_confirmation: marking an intermediate result or branch as now settled; example words: \"now we know\", “so we’ve determined\"&#x5c;n"
    "answer_reporting: presentation of the final answer with no further reasoning; example words: \"final answer:\", \"result:\""
)

THINKING_BUDGET = 600  # as requested: 500–600 tokens

###########################################################################
# Pydantic Models for Response Parsing                                     #
###########################################################################

class SentenceAnnotation(BaseModel):
    """One sentence and up to two categories."""

    sentence: str
    categories: List[str] = Field(min_items=1, max_items=2)

    # Lightweight validation (Gemini already filtered, but be safe)
    @computed_field
    @property
    def _validate_categories(self) -> None:
        invalid = [c for c in self.categories if c not in CATEGORY_DEFINITIONS]
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}")
        return None

class CoTAnnotation(BaseModel):
    question_id: int
    sentence_annotations: Dict[str, SentenceAnnotation]

###########################################################################
# Utility Functions                                                        #
###########################################################################

def _read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(data, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _split_completion(raw_completion: str) -> tuple[str, str]:
    """Return (user_input, model_cot) from the raw completion string.

    The heuristic mirrors the earlier script.  It looks for the sentinel
    ``<|end_header_id|>`` marking the assistant section, and for the optional
    closing ``</think>`` tag.  If either is missing we fall back gracefully.
    """
    lower = raw_completion.lower()

    # Where does assistant reasoning start?
    assistant_idx = lower.find("<|end_header_id|>")
    if assistant_idx == -1:
        logging.warning("<|end_header_id|> not found – falling back to 'assistant'")
        assistant_idx = lower.find("assistant")
        if assistant_idx == -1:
            raise ValueError("Could not locate assistant section in completion string")

    # Where does the \</think> tag end?
    think_end = lower.rfind("</think>")
    if think_end == -1:
        logging.info("No </think> tag – using end of string as CoT end marker")
        think_end = len(raw_completion)

    user_input = raw_completion[:assistant_idx]
    model_cot = raw_completion[assistant_idx:think_end]
    return user_input.strip(), model_cot.strip()

###########################################################################
# LLM Call                                                                 #
###########################################################################

def _annotate_with_gemini(question_id: int, cot_text: str, client: genai.Client) -> CoTAnnotation:
    """Call Gemini to annotate *one* CoT and return a validated `CoTAnnotation`."""

    prompt = f"""You are a meticulous verifier.  Your job is to read the following Chain-of-Thought (CoT) produced by an LLM and categorise *each individual sentence* according to **exactly** one of the predefined phrase-type categories listed below (two categories only if a sentence genuinely belongs to two).  Use the categories *verbatim* and do **not** invent new ones.  After categorising, output **only** a JSON object in the exact structure given.

Each chain-of-thought text must be split up into distinct phrase categories:
{CATEGORY_DEFINITIONS}

Carefully think about the meaning of each category.  Then split the CoT into sentences and assign the right category to each.  If a sentence *must* belong to two categories, list both in the same order they occur in the list above.

Output format (JSON *only* – no markdown):
{{
  "question_id": {question_id},
  "sentence_annotations": {{
    "1": {{"sentence": "<first sentence>", "categories": ["<category>"]}},
    "2": {{"sentence": "<second sentence>", "categories": ["<category>"]}},
    ...
  }}
}}

Chain-of-Thought:
""" + cot_text

    generate_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        response_mime_type="application/json",
        response_schema=CoTAnnotation,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
        config=generate_config,
    )

    return response.parsed  # Already validated by pydantic

###########################################################################
# Public Runner                                                            #
###########################################################################

def run_cot_annotation(
    dataset_name: str,
    hint_types: List[str],
    model_name: str,
    n_questions: int,
    *,
    api_key: str | None = None,
) -> None:
    """Annotate all CoTs in *data/<dataset>/<model>/<hint_type>*.

    Parameters
    ----------
    dataset_name, hint_types, model_name, n_questions
        Folder structure parameters mirroring the old script.
    api_key
        Explicit API key; if ``None``, the environment variable ``GOOGLE_API_KEY``
        (or ``GEMINI_API_KEY``) is used.
    """

    # Initialise Gemini client once
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("No API key supplied (env GOOGLE_API_KEY or GEMINI_API_KEY)")
    client = genai.Client(api_key=key)

    for hint_type in hint_types:
        print(f"\n▶ Processing hint type: {hint_type}")

        # Paths – re-use naming convention of previous pipeline
        base = Path("data") / dataset_name / model_name / hint_type
        completions_path = base / f"completions_with_{n_questions}.json"
        output_path = base / f"cot_annotations_with_{n_questions}.json"

        completions = _read_json(completions_path)
        out: List[dict] = []

        for comp in completions:
            qid = comp["question_id"]
            try:
                _, cot = _split_completion(comp["completion"])
                annotation = _annotate_with_gemini(qid, cot, client)
                out.append(annotation.model_dump(mode="json"))
            except Exception as exc:
                logging.exception(
                    "Failed to annotate question_id=%s – storing error entry", qid
                )
                out.append({
                    "question_id": qid,
                    "error": str(exc),
                })

        _save_json(out, output_path)
        print(f"✔ Saved annotations to {output_path}")

###########################################################################
# Convenience single-file runner                                           #
###########################################################################

def annotate_file(
    completions_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
) -> None:
    """Simpler helper when you just have a single completions JSON file."""

    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    completions = _read_json(completions_path)
    out = []

    for comp in completions:
        qid = comp["question_id"]
        _, cot = _split_completion(comp["completion"])
        anno = _annotate_with_gemini(qid, cot, client)
        out.append(anno.model_dump(mode="json"))

    _save_json(out, output_path)
    print(f"✔ Wrote annotations to {output_path}")
