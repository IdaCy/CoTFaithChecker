# reasoning_category_discovery.py
"""
Discover a mutually-exclusive, collectively-exhaustive set of reasoning categories
from LLM chain-of-thought completions.

USAGE
-----
python reasoning_category_discovery.py \
       --files data/mmlu/DeepSeek-R1-Distill-Llama-8B/none/completions_with_500.json \
               data/mmlu/DeepSeek-R1-Distill-Llama-8B/sycophancy/completions_with_500.json \
       --max-cats 12 \
       --out categories.json
"""

import argparse, json, os, random, textwrap
from typing import List
from tqdm import tqdm
#from google import genai
from google.genai import types

from a_confirm_posthoc.eval.llm_hint_verificator import client

# -------- helpers -------------------------------------------------------------

def load_completions(paths: List[str]) -> List[str]:
    completions = []
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        completions += [d["completion"] for d in data]
    return completions

def build_prompt(examples: List[str], max_cats: int) -> str:
    examples_txt = "\n###\n".join(textwrap.shorten(e, 2000, placeholder=" ...")   # keep prompt short
                                   for e in examples)
    return f"""
You are an expert analyst of large language-model chain-of-thought (CoT) traces.
Below are EXCERPTS of CoTs separated by ###.  
Derive **a set of high-level reasoning categories** that are:

* mutually exclusive (no overlap)  
* collectively exhaustive (every token in *any* CoT fits one & only one category)  
* phrased in concise `snake_case`
* include e.g. "problem-understanding", "optimization", "back-tracking", "intermediate-calculation", "step-by-step-solution", "intermediate-calculation", "compare-to-options", "rephrasing-or-summarizing", "identifying-key-information", "formula-application", "evaluating-scenarios", "checking-conditions", "interpreting-results"

Return **ONLY** a JSON array (e.g. `["problem_understanding", "backtracking", ...]`)
with at most {max_cats} elements.

CoT examples:
###
{examples_txt}
###
"""

def discover_categories(file_paths: List[str],
                        max_cats: int = 12,
                        sample_size: int = 30) -> List[str]:
    #client = genai.Client()
    completions = load_completions(file_paths)
    sample = random.sample(completions, min(sample_size, len(completions)))

    prompt = build_prompt(sample, max_cats)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )
    cats = response.text
    try:
        cats = json.loads(cats)
        assert isinstance(cats, list) and all(isinstance(c, str) for c in cats)
    except Exception as e:
        raise ValueError(f"Could not parse categories:\n{response.text}") from e
    return cats

# -------- cli -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Discover CoT reasoning categories.")
    ap.add_argument("--files", nargs="+", required=True, help="List of completion JSONs")
    ap.add_argument("--max-cats", type=int, default=12)
    ap.add_argument("--out", default="categories.json")
    args = ap.parse_args()

    cats = discover_categories(args.files, args.max_cats)
    with open(args.out, "w") as f:
        json.dump(cats, f, indent=2)
    print("Categories written to", args.out)
    print(cats)

if __name__ == "__main__":
    main()
