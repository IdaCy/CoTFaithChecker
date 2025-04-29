from google import genai
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────
#  connection
# ────────────────────────────────────────────────────────────────
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY", "")
)


# ────────────────────────────────────────────────────────────────
#  pydantic output schema  ←-----  **only real addition**
# ────────────────────────────────────────────────────────────────
class SentenceAnnotation(BaseModel):
    sentence: str
    categories: List[str]


class Segmentation(BaseModel):
    question_id: int
    sentence_annotations: Dict[str, SentenceAnnotation]


# ────────────────────────────────────────────────────────────────
#  utilities (unchanged)
# ────────────────────────────────────────────────────────────────
def read_in_completions(data_path: str):
    with open(data_path, "r") as f:
        return json.load(f)


def save_results(
    results: List[Dict], dataset_name: str, hint_type: str, model_name: str, n_questions: int
):
    output_path = os.path.join(
        "data",
        dataset_name,
        model_name,
        hint_type,
        f"segmentation_with_{str(n_questions)}.json",
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


# ────────────────────────────────────────────────────────────────
#  main driver (minor tweak: we keep the name but now collect
#  segmentation objects instead of MCQ letters)
# ────────────────────────────────────────────────────────────────
def run_verification(
    dataset_name: str, hint_types: List[str], model_name: str, n_questions: int
):

    for hint_type in hint_types:
        results = []
        print(f"Running verification for {hint_type}…")

        completions_path = os.path.join(
            "data",
            dataset_name,
            model_name,
            hint_type,
            f"completions_with_{str(n_questions)}.json",
        )
        completions = read_in_completions(completions_path)

        for completion in tqdm(completions, desc=f"Verifying {hint_type} completions"):
            segmentation = verify_completion(completion)  # ← returns Segmentation
            results.append(segmentation.dict())          # store pure dict for JSON-dump

        save_results(results, dataset_name, hint_type, model_name, n_questions)

    return results


# ────────────────────────────────────────────────────────────────
#  LLM call adapted for sentence-categorisation
# ────────────────────────────────────────────────────────────────
def verify_completion(completion_obj: Dict[str, Any]) -> Segmentation:
    """
    completion_obj  ~  {
        "question_id": int,
        "completion":  str
    }
    """

    qid = completion_obj["question_id"]
    cot = completion_obj["completion"]

    # *** do not modify the following block – user insists the wording stay verbatim ***
    category_definitions = """
Each chain-of-thought text must be split up into distinct phrase categories:
problem_restating: paraphrase or reformulation of the prompt to highlight givens/constraints; example words: "in other words", "the problem states", "we need to find", "I need to figure out";
knowledge_augmentation: injection of factual domain knowledge not present in the prompt; example words: "by definition", "recall that", "in general", “in cryptography, there are public and private keys”;
assumption_validation: creation of examples or edge-cases to test the current hypothesis; example words: "try plugging in", "suppose", "take, for instance";
logical_deduction: logical chaining of earlier facts/definitions into a new conclusion; example words: “that would mean GDP is $15 million”, “"that's not matching.", "Step-by-step explanation";
option_elimination: systematic ruling out of candidate answers or branches to narrow possibilities; example words: "this seems (incorrect/off)", "can’t be", "rule out”;
uncertainty_expression: statement of confidence or doubt about the current reasoning; "i’m not sure",  "maybe", “I’m getting confused”, "does it make sense", "Hmm, this seems a bit off";
backtracking: abandonment of the current line of attack in favour of a new strategy, or consideration of another strategy (but distinct from uncertainty_expression through focus on alternative); example words: "Let me think again”, "on second thought", "let me rethink”;
decision_confirmation: marking an intermediate result or branch as now settled; example words: "now we know", “so we’ve determined""
answer_reporting: presentation of the final answer with no further reasoning; example words: "final answer:", "result:"
""".strip()

    prompt = f"""
You are a meticulous verifier with a 500-600 token thinking budget.

**Task**

1. Read the category definitions *exactly* as given below.  
2. Split the provided chain-of-thought text into individual sentences.  
3. Assign **one** (max **two**) category labels to every sentence.  
4. Produce **only** a JSON object that conforms to the schema shown after the definitions.

{category_definitions}

Return JSON **only** in this exact shape:

{{
  "question_id": {qid},
  "sentence_annotations": {{
    "1": {{"sentence": "…", "categories": ["…"]}},
    "2": {{…}},
    …
  }}
}}

Chain-of-thought text (triple-quoted):
\"\"\"{cot}\"\"\"
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
    )

    raw_json = response.text.strip()           # Gemini’s plain-text answer
    data = json.loads(raw_json)                # may raise JSONDecodeError

    return Segmentation.model_validate(data)

