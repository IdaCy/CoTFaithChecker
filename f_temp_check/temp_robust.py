import torch
import sys # Import sys
import json
import os
import logging
import argparse
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import time
from typing import Dict, List, Tuple, Optional, Any


# --- Project Specific Imports ---
# Assuming PYTHONPATH is set correctly or script is run from workspace root
try:
    from a_confirm_posthoc.utils.model_handler import load_model_and_tokenizer
except ImportError:
    print(os.getcwd())
    logging.error("Failed to import from a_confirm_posthoc.utils.model_handler. Ensure PYTHONPATH is set or run from project root.")
    exit(1)
# We will import verification functions in Phase 2

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_json(file_path: str) -> Any:
    """Loads JSON data from a file."""
    logging.info(f"Loading JSON from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {file_path}: {e}")
        raise

def save_json(data: Any, file_path: str):
    """Saves data to a JSON file."""
    logging.info(f"Saving JSON to: {file_path}")
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"An unexpected error occurred saving to {file_path}: {e}")
        raise

def extract_prompt_text(completion_string: str) -> Optional[str]:
    """Extracts the user prompt part ending before the assistant's turn."""
    markers = ["<|start_header_id|>assistant<|end_header_id|>\n", "<|im_start|>assistant\n"]
    min_index = float('inf')
    found_marker = False
    found_marker_text = ""

    for marker in markers:
        try:
            index = completion_string.find(marker)
            if index != -1:
                if index < min_index:
                    min_index = index
                    found_marker = True
                    found_marker_text = marker
        except Exception: # Catch potential errors if marker isn't string etc.
             pass

    if found_marker:
        # Return everything up to and including the assistant marker
        if found_marker_text in markers:
            return completion_string[:min_index + len(found_marker_text)]
        else:
            return completion_string[:min_index]
    else:
        logging.warning("Could not find standard assistant marker in completion string.")
        # Fallback or further logic could be added here if needed
        return None # Or maybe return the whole string with a warning?

def parse_hint_option(hint_text: str) -> Optional[str]:
    """Extracts the hinted option (e.g., 'A', 'B') from hint text."""
    # Regex to find patterns like "[ A ]", "[B]", etc. (case-insensitive)
    match = re.search(r'\[\s*([A-Da-d])\s*\]', hint_text)
    if match:
        return match.group(1).upper()
    else:
        logging.warning(f"Could not parse hint option from text: {hint_text}")
        return None

def generate_n_completions_batched(
    prompts_data: List[Dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    num_generations: int,
    temperature: float,
    max_new_tokens: int,
    batch_size: int
) -> Dict[int, List[str]]:
    """
    Generates N completions for each prompt using batching.
    Returns a dictionary {question_id: [list_of_N_completions]}.
    """
    results_store = {item['question_id']: [] for item in prompts_data}
    prompt_map = {item['question_id']: item['prompt_text'] for item in prompts_data}
    question_ids_ordered = [item['question_id'] for item in prompts_data]

    logging.info(f"Starting generation of {num_generations} completions for {len(prompts_data)} questions...")

    for n in range(num_generations):
        logging.info(f"--- Generating run {n+1}/{num_generations} ---")
        # Process in batches
        for i in tqdm(range(0, len(question_ids_ordered), batch_size), desc=f"Batch for Run {n+1}"):
            batch_qids = question_ids_ordered[i:i + batch_size]
            batch_prompts = [prompt_map[qid] for qid in batch_qids]

            # Tokenize the batch
            # Ensure padding side is handled correctly by tokenizer/model loading
            encodings = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt", max_length=model.config.max_position_embeddings - max_new_tokens - 10) # Added max_length safeguard
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Generate
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id # Crucial for open-ended generation with padding
                        # top_p=0.95 # Added common sampling parameter
                    )

                # Decode full sequence
                decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False) # Keep special tokens for now
                

                # Store results
                for qid, decoded_text in zip(batch_qids, decoded_texts):
                    # Basic cleanup: remove padding token string if tokenizer adds it explicitly at the end
                    # Note: skip_special_tokens=False often handles this, but some tokenizers might behave differently.
                    cleaned_text = decoded_text.replace(tokenizer.pad_token, "").strip()
                    #find the <|end_header_id|>\n and insert <think>\n after it
                    try:
                        cleaned_text = re.sub(r'<\|end_header_id\|>\n', r'<|end_header_id|>\n<think>\n', cleaned_text)
                    except:
                        cleaned_text = re.sub(r'<|im_start|>assistant\n"', r'<|im_start|>assistant\n"<think>\n', cleaned_text)
                    # Further cleanup might be needed depending on model/tokenizer artifacts
                    results_store[qid].append(cleaned_text)

            except Exception as e:
                 logging.error(f"Error during generation for batch starting with QID {batch_qids[0]} in run {n+1}: {e}")
                 # Store error marker or skip? For now, let's add error markers
                 for qid in batch_qids:
                    results_store[qid].append(f"GENERATION_ERROR: {e}")

            # Optional: Clear cache between batches if memory pressure is high
            # torch.cuda.empty_cache()

    return results_store


# --- Main Execution Logic ---
def run_generation_phase(args):
    """Runs the generation phase."""
    logging.info("--- Starting Generation Phase ---")

    # --- 1. Setup ---
    # Construct model name from path for output files
    model_name_suffix = args.model_path.split("/")[-1]
    output_dir = args.output_dir or os.path.join("f_temp_check", "outputs", args.dataset_name, model_name_suffix, args.hint_type)
    os.makedirs(output_dir, exist_ok=True)
    raw_output_file = os.path.join(output_dir, f"temp_generations_raw_{args.dataset_name}_{args.n_questions}.json")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Raw output file: {raw_output_file}")

    # --- 2. Load Model ---
    try:
        model, tokenizer, loaded_model_name, device = load_model_and_tokenizer(args.model_path)
        # Ensure model_name_suffix matches loaded name if needed, though suffix from path is usually fine
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return # Exit generation phase

    # --- 3. Load Input Data ---
    try:
        hint_verification_path = os.path.join("data", args.dataset_name, model_name_suffix, args.hint_type, f"hint_verification_with_{args.n_questions}.json")
        target_questions_data = load_json(hint_verification_path) # List of dicts with question_id, verbalizes_hint, etc.

        completions_path = os.path.join("data", args.dataset_name, model_name_suffix, args.hint_type, f"completions_with_{args.n_questions}.json")
        original_completions = load_json(completions_path) # List of dicts with question_id, completion
        original_completions_map = {item['question_id']: item['completion'] for item in original_completions}

        # We need hint_option. Try getting it from switch_analysis first as it might be cleaner
        switch_analysis_path = os.path.join("data", args.dataset_name, model_name_suffix, args.hint_type, f"switch_analysis_with_{args.n_questions}.json")
        switch_analysis_data = load_json(switch_analysis_path) # List of dicts with question_id, hint_option
        hints_options_map = {item['question_id']: item['hint_option'] for item in switch_analysis_data}

    except Exception as e:
        logging.error(f"Failed to load necessary input data files: {e}")
        return

    # --- 4. Prepare Prompts & Target Info ---
    prompts_for_generation = []
    target_info = {}
    questions_processed_count = 0

    logging.info("Preparing prompts...")
    for item in target_questions_data:
        qid = item['question_id']
        original_verbalizes = item.get('verbalizes_hint', None) # Safely get value

        if qid not in original_completions_map:
            logging.warning(f"QID {qid} from hint verification not found in original completions. Skipping.")
            continue
        if qid not in hints_options_map:
            logging.warning(f"Hint option for QID {qid} not found in switch analysis. Skipping.")
            continue

        original_completion_text = original_completions_map[qid]
        prompt_text = extract_prompt_text(original_completion_text)
        if prompt_text is None:
            logging.warning(f"Could not extract prompt for QID {qid}. Skipping.")
            continue

        hint_option = hints_options_map[qid]

        prompts_for_generation.append({'question_id': qid, 'prompt_text': prompt_text})
        target_info[qid] = {'original_verbalizes_hint': original_verbalizes, 'hint_option': hint_option}

        questions_processed_count += 1
        if args.demo_mode_limit is not None and questions_processed_count >= args.demo_mode_limit:
            logging.info(f"Reached demo mode limit of {args.demo_mode_limit} questions.")
            break

    if not prompts_for_generation:
        logging.error("No valid prompts prepared for generation. Exiting.")
        return

    logging.info(f"Prepared {len(prompts_for_generation)} prompts for generation.")

    # --- 5. Generate Completions ---
    start_time = time.time()
    generated_data = generate_n_completions_batched(
        prompts_data=prompts_for_generation,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    end_time = time.time()
    logging.info(f"Generation finished in {end_time - start_time:.2f} seconds.")

    # --- 6. Structure and Save Raw Results ---
    raw_results_list = []
    for qid, generations_list in generated_data.items():
        if qid in target_info: # Ensure we only save results for prompts we prepared
            raw_results_list.append({
                'question_id': qid,
                'original_verbalizes_hint': target_info[qid]['original_verbalizes_hint'],
                'hint_option': target_info[qid]['hint_option'],
                'generations': generations_list
            })
        else:
             logging.warning(f"Generated data for QID {qid} but it wasn't in the initial target info. Ignoring.")

    config_data = vars(args) # Save command line args used

    full_raw_output = {'config': config_data, 'raw_generations': raw_results_list}
    save_json(full_raw_output, raw_output_file)
    logging.info(f"Raw generation results saved to {raw_output_file}")

    # --- 7. Cleanup ---
    logging.info("Cleaning up model and GPU memory...")
    del model
    del tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    logging.info("Generation phase complete.")

def run_analysis_phase(args):
    """Runs the analysis phase."""
    logging.info("--- Starting Analysis Phase ---")
    # Placeholder for Phase 2 implementation
    # ... load raw data ...
    # ... initialize/load detailed analysis file ...
    # ... loop through questions/generations ...
    # ... call verification functions ...
    # ... save detailed results incrementally ...
    # ... calculate summaries ...
    # ... save summary ...
    # ... print results ...
    logging.warning("Analysis phase not yet implemented.")
    pass




# Default configuration
config = {
    "model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # Example model path
    "dataset_name": "mmlu_pro",
    "hint_type": "sycophancy",
    "n_questions": 300,
    "output_dir": "f_temp_check/outputs",
    "demo_mode_limit": 2,  # Set to None to process all questions
    "num_generations": 3,
    "temperature": 0.7,
    "max_new_tokens": 1000,
    "batch_size": 8
}

# Create a simple args object to pass to the functions
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

args = Args(**config)

# Uncomment the function you want to run
run_generation_phase(args)
# run_analysis_phase(args)

logging.info("Script finished.")
