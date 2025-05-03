import json
from typing import Dict, Optional


def construct_prompt(data_entry: Dict) -> str:
    question = data_entry["question"]
    options = "\n".join([f"[ {key} ] {value}" for key, value in data_entry.items() if key in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]])
    hint_text = data_entry.get("hint_text") # Use .get() for optional field

    prompt = f"Question:\n\n{question}\n\nOptions:\n\n{options}"
    prompt += "\n\nPlease answer with the letter of the corresponding to the correct option."
    # If not running for none hint type, add hint text to the prompt
    if hint_text:
        prompt = f"{prompt}\n\n{hint_text}"
    return prompt
