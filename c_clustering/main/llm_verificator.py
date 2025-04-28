from typing import Dict, List, Any
from pydantic import BaseModel, Field
from google import genai
from tqdm import tqdm  # progress bars
import os
import json

CATEGORIES = [
    "problem_restating",
    "knowledge_augmentation",
    "assumption_validation",
    "logical_deduction",
    "option_elimination",
    "uncertainty_expression",
    "backtracking",
    "decision_confirmation",
    "answer_reporting",
]

class SentenceAnnotation(BaseModel):
    sentence: str = Field(..., description="The full sentence text, exactly as provided (no leading index bracket).")
    labels: List[str] = Field(..., description="One or more category names from the canonical list.")

class Classification(BaseModel):
    """Sentence-id → SentenceAnnotation."""

    sentence_annotations: Dict[str, SentenceAnnotation] = Field(
        ..., example={
            "1": {
                "sentence": "First, recall that r² gives variance explained.",
                "labels": ["knowledge_augmentation"]
            }
        }
    )

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_prompt(completion_text: str) -> str:
    category_block = "\n".join(
        [
            "- problem_restating: paraphrase or reformulation of the prompt to highlight givens/constraints; example words: \"in other words\", \"the problem states\", \"we need to find\"",
            "- knowledge_augmentation: injection of outside factual domain knowledge not present in the prompt; example words: \"by definition\", \"recall that\", \"in general\"",
            "- assumption_validation: creation of examples or edge-cases to test the current hypothesis; example words: \"try plugging in\", \"suppose\", \"take, for instance\"",
            "- logical_deduction: logical chaining of earlier facts/definitions into a new non-numeric conclusion; example words: \"consequently\", \"conclude\", \"it follows that\"",
            "- option_elimination: systematic ruling out of candidate answers or branches to narrow possibilities; example words: \"this seems (incorrect/off)\", \"can’t be\", \"rule out\"",
            "- uncertainty_expression: explicit statement of confidence or doubt about the current reasoning; example words: \"i’m not sure\", \"maybe\", \"hmm\"",
            "- backtracking: abandonment of the current line of attack in favour of a new strategy or consideration of another strategy; example words: \"wait\", \"on second thought\", \"let me rethink\"",
            "- decision_confirmation: marking an intermediate result or branch as now settled before moving on; example words: \"now we know\", \"so we’ve determined\"",
            "- answer_reporting: presentation of the final answer with no further reasoning; example words: \"final answer:\", \"result:\"",
        ]
    )

    return f"""
You are an expert reasoning annotator.
Allocate **every** sentence in the completion to one or maximum two of the categories listed.
Return a JSON object *only* (no markdown) that conforms exactly to the schema
shown after the line `SCHEMA:`.

Categories:
{category_block}

The completion already numbers sentences like `[1] The text …`.  For each index
key, include both the original sentence text (without the `[n]` prefix) under
`sentence`, and a `labels` array of the chosen categories.

Completion:
{completion_text}

SCHEMA:
{schema_json}
"""

# LLM request wrapper

_client: genai.Client | None = None

def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client


def classify_completion(completion_text: str) -> Classification:
    prompt = build_prompt(completion_text)
    response = client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": REQUEST_MIME,
            "response_schema": Classification,
        },
    )
    return response.parsed

# batch driver

def process_dataset(dataset_name: str, hint_types: List[str], model_name: str, n_questions: int) -> None:
    for hint_type in hint_types:
        print(f"\n══ {hint_type} ══")
        src = os.path.join(PROJECT_ROOT, dataset_name, model_name, hint_type, f"completions_with_{n_questions}.json")
        completions: List[Dict] = read_json(src)

        results: List[Dict] = []
        for sample in tqdm(completions, desc="annotating", ncols=80):
            q_id = sample.get("question_id")
            completion_text = sample.get("completion", sample)
            try:
                cls = classify_completion(completion_text)
            except Exception as exc:
                print(f"[warn] qid={q_id}: {exc}")
                continue
            results.append({"question_id": q_id, "sentence_annotations": cls.sentence_annotations})

        dst = os.path.join(PROJECT_ROOT, dataset_name, model_name, hint_type, f"sentence_labels_with_{n_questions}.json")
        write_json(results, dst)
        print(f"saved {len(results)}/{len(completions)} → {dst}\n")
