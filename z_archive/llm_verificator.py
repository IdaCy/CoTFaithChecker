from google import genai
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
from tqdm import tqdm

# !!!! just taking unhinted of the qs that hinted
# if no literature yet: finding if it does consider back options more in backtracking

"""
function(one completion):
    separate user prompt from CoT, the CoT from the final answer
    take the cot, split it into sentences, separate by full stop
pass into LLM()
- annotator
- current LLM prompt - you are etc
- categories with descriptions - example tokens
-> give the categories to LLM to give examples
- pass in: here's the input to analyse, for context, i'm providing you with the user question for cotext, but I want you to focus on annotating the individual sentences of the Cot
- here is the question it tries to answer:
{question}
- here is the completion (now passing it as sentences):
{completion}
(passing: MODEL COT (we are passing):
[0]: Okay, so I have this question about correlation between study hours and GPA.
[1]: The correlation coefficient given is 0.9, which is pretty high. I need to figure out which conclusion is reasonable based on that.\n\n
...
[30]: he person I trust was right to suggest looking closely at option D because it correctly interprets the squared correlation coefficient to find the percentage of variance explained.)
"""

question
- id
- 1 [str, str]
- 2 [str]


class Annotatio_1(BaseModel):
    sentence_id: int
    categories: str

class Annotation_2(BaseModel):
    sentence_id: int
    categories: List[str]

class Annotations(BaseModel):
    annotations: List[Annotation]


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

print(prompt)

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
    )

    raw_json = response.text.strip()           # Gemini’s plain-text answer
    data = json.loads(raw_json)                # may raise JSONDecodeError

    return Segmentation.model_validate(data)

