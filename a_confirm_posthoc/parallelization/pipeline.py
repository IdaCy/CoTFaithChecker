import json, os, time, logging
from typing import List, Dict, Optional

import torch
from a_confirm_posthoc.utils.prompt_constructor import construct_prompt
from a_confirm_posthoc.parallelization.model_handler import generate_completion
from accelerate.utils import gather_object

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

KNOWN_CHAT_TEMPLATES = {
    "llama":  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "qwen":   "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
}

def get_chat_template(model_name: str) -> str:
    name = model_name.lower()
    if "llama" in name:
        return KNOWN_CHAT_TEMPLATES["llama3"]
    if "qwen" in name:
        return KNOWN_CHAT_TEMPLATES["qwen"]
    logging.warning(f"No specific chat template for {model_name}; using llama.")
    return KNOWN_CHAT_TEMPLATES["llama"]

# helpers
def load_data(path: str) -> List[Dict]:
    if not os.path.exists(path):
        logging.error(f"Data file not found: {path}")
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        return []

def save_results(results: List[Dict], dataset: str, hint: str,
                 model: str, n_q: int) -> None:
    out = os.path.join("data", dataset, model, hint,
                       f"completions_with_{n_q}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {out}")

# main
def generate_dataset_completions(
    accelerator,
    model, tokenizer, model_name, device,
    dataset_name: str,
    hint_types: List[str],
    batch_size: int = 8,
    max_new_tokens: Optional[int] = 512,
    n_questions: Optional[int] = None,
) -> None:

    start = time.time()
    if accelerator.is_main_process:
        logging.info(f"Running on {accelerator.num_processes} GPU(s)")

    chat_template = get_chat_template(model_name)

    for hint_type in hint_types:
        logging.info(f"--- Processing hint type: {hint_type} ---")

        q_path = os.path.join("data", dataset_name, "input_mcq_data.json")
        h_path = os.path.join("data", dataset_name, f"hints_{hint_type}.json")

        data  = load_data(q_path)[:n_questions]
        hints = load_data(h_path)[:n_questions]

        rank, world = accelerator.process_index, accelerator.num_processes
        data  = data [rank::world]
        hints = hints[rank::world]

        # merge hints into questions
        hint_dict = {h["question_id"]: h for h in hints}
        for entry in data:
            h = hint_dict.get(entry["question_id"])
            entry["hint_text"] = h.get("hint_text") if h else None

        # build prompts
        prompts = []
        for entry in data:
            prompt = construct_prompt(entry).rstrip() + "\n<think>"
            prompts.append({"question_id": entry["question_id"],
                            "prompt_text": prompt})

        # generate
        results = generate_completion(
            model, tokenizer, device, prompts,
            batch_size, max_new_tokens
        )

        # gather & save
        #gathered = accelerator.gather_object(results)
        accelerator.wait_for_everyone()
        gathered = gather_object(results)           # list[ list[dict] ] on rank-0
        if accelerator.is_main_process:
            merged = [d for lst in gathered for d in (lst if isinstance(lst, list) else [lst])]
        #gathered = gather_object(results)
        #if accelerator.is_main_process:
        #    merged = [item for sub in gathered for item in sub]
            save_results(merged, dataset_name, hint_type, model_name, n_questions)

    if accelerator.is_main_process:
        logging.info(f"Total time: {time.time() - start:.2f} s")

if __name__ == "__main__":
    pass