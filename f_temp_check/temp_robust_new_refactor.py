import torch
import sys # Import sys
import json
import os
import logging
import argparse
from accelerate import Accelerator # Add Accelerator
from accelerate.utils import gather_object # Add gather_object
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict # Add defaultdict import
import gc

# accelerate launch f_temp_check/temp_robust_new_refactor.py


# --- Project Specific Imports ---
# Assuming PYTHONPATH is set correctly or script is run from workspace root
# Remove the old model_handler import
# try:
#     from a_confirm_posthoc.utils.model_handler import load_model_and_tokenizer
# except ImportError:
#     print(os.getcwd())
#     logging.error("Failed to import from a_confirm_posthoc.utils.model_handler. Ensure PYTHONPATH is set or run from project root.")
#     exit(1)

# Import verification functions with aliases
try:
    from a_confirm_posthoc.eval.llm_verificator import verify_completion as extract_final_answer_llm
    from a_confirm_posthoc.eval.llm_verificator import Verification as AnswerVerificationResult
    from a_confirm_posthoc.eval.llm_hint_verificator import verify_completion as verify_hint_details_llm
    from a_confirm_posthoc.eval.llm_hint_verificator import Verification as HintVerificationResult
    from a_confirm_posthoc.eval.llm_hint_verificator import split_completion # If needed directly
except ImportError:
    logging.error("Failed to import verification functions from a_confirm_posthoc.eval. Ensure PYTHONPATH is set.")
    exit(1) # Keep exit here, critical dependency

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(process)d - %(message)s') # Added process ID

# --- Initialize Accelerator ---
accelerator = Accelerator()

# Setup logging to be more informative and less verbose on non-main processes
if not accelerator.is_main_process:
    logging.getLogger().setLevel(logging.INFO)  # show INFO from all ranks
else:
    # Add stream handler only for main process to see logs in console
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Accelerator initialized on {accelerator.num_processes} processes.")


# --- Helper Functions ---

def load_json(file_path: str) -> Any:
    """Loads JSON data from a file."""
    # Use accelerator.is_main_process to avoid redundant logging from all processes
    if accelerator.is_main_process:
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
    # Saving should only happen on the main process
    if accelerator.is_main_process:
        logging.info(f"Saving JSON to: {file_path}")
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"An unexpected error occurred saving to {file_path}: {e}")
            # Don't raise here, let the main process handle reporting
    else:
        # Maybe log a debug message if needed, but generally do nothing on non-main processes
        pass

def extract_prompt_text(completion_string: str) -> Optional[str]:
    """Extracts the user prompt part ending before the assistant's turn."""
    markers = ["<｜Assistant｜><think>\n"]
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
    Generates N completions for each prompt using batching, distributed across GPUs.
    Each process returns a dictionary {question_id: [list_of_N_completions]} for its subset of prompts.
    """
    # Use accelerator device
    device = accelerator.device
    # Access underlying model if necessary (needed for generate)
    gen_model = model.module if hasattr(model, 'module') else model
    # Each process initializes its own results store for its slice of data
    local_results_store = {item['question_id']: [] for item in prompts_data}
    prompt_map = {item['question_id']: item['prompt_text'] for item in prompts_data}
    question_ids_ordered = [item['question_id'] for item in prompts_data]

    num_local_prompts = len(prompts_data)
    if num_local_prompts == 0:
        logging.warning(f"Process {accelerator.process_index} received no prompts.")
        return {}

    logging.info(f"Process {accelerator.process_index}: Starting generation of {num_generations} completions for {num_local_prompts} questions...")

    for n in range(num_generations):
        # Only show overall run progress on main process
        if accelerator.is_main_process:
             logging.info(f"--- Generating run {n+1}/{num_generations} ---")

        # Use tqdm only on the main process for cleaner logs
        # The iterator logic remains the same, processing local batches
        iterable = range(0, num_local_prompts, batch_size)
        if accelerator.is_main_process:
            iterable = tqdm(iterable, desc=f"Batch for Run {n+1} (Main Process)")
        else:
            iterable = tqdm(iterable, desc=f"Batch for Run {n+1} (Process {accelerator.process_index})", disable=not accelerator.is_local_main_process) # Show tqdm per machine

        for i in iterable:
            batch_qids = question_ids_ordered[i:i + batch_size]
            batch_prompts = [prompt_map[qid] for qid in batch_qids]

            # Tokenize the batch - tokenizer is already prepared by accelerator
            encodings = tokenizer(batch_prompts,
            padding=True,
            truncation=False, # Important not to truncate prompts
            return_tensors="pt").to(device) # Move tensors to the correct device

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            # Generate using the potentially wrapped model
            try:
                with torch.no_grad():
                    input_length = input_ids.shape[1]
                    # Use gen_model for generation
                    outputs = gen_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.pad_token_id # Use the prepared tokenizer's pad_token_id
                        # top_p=0.95 # Added common sampling parameter
                    )

                # Decode only the generated part and reconstruct
                # outputs are already on the correct device
                generated_ids = outputs[:, input_length:]
                # Use the prepared tokenizer for decoding
                generated_parts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for j, qid in enumerate(batch_qids):
                    original_prompt = batch_prompts[j]
                    generated_part = generated_parts[j]

                    # Construct the final string
                    final_completion = original_prompt + generated_part.strip() + tokenizer.eos_token

                    local_results_store[qid].append(final_completion)

            except Exception as e:
                 # Log error specific to the process encountering it
                 logging.error(f"Process {accelerator.process_index}: Error during generation for batch starting with QID {batch_qids[0]} in run {n+1}: {e}")
                 for qid in batch_qids:
                    local_results_store[qid].append(f"GENERATION_ERROR: {e}")

            # Optional: Clear cache - less critical with accelerate managing memory potentially
            # torch.cuda.empty_cache()

    # Each process returns its dictionary of results
    return local_results_store

def generate_completion_accelerate(
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    batch_size: int = 8,
    max_new_tokens: Optional[int] = 512,
    temperature: float = 0.7,
):
    """Generate completions for a list of prompt dicts using Accelerate-prepared model.

    Each *prompt* element must be a dict with keys:
        - question_id : str
        - prompt_text : str (already has any chat template applied)

    Returns a *local* list of dicts [{"question_id": id, "completion": text}, ...]
    Calling code is responsible for gather_object on these lists.
    """

    results: List[Dict[str, str]] = []
    gen_max = max_new_tokens or 2048
    logging.info(f"[generate_completion_accelerate] Using max_new_tokens={gen_max}")

    gen_model = model.module if hasattr(model, "module") else model
    gen_device = next(gen_model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [p["prompt_text"] for p in batch]
        qids = [p["question_id"] for p in batch]

        logging.info(
            f"[rank {accelerator.process_index}] Batch {i//batch_size+1}/"
            f"{(len(prompts)+batch_size-1)//batch_size} (size {len(texts)})")

        enc = tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(gen_device)
        attention_mask = enc["attention_mask"].to(gen_device)
        inp_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Wrap with explicit BOS/EOS tokens like model_handler.generate_completion
        decoded = [tokenizer.bos_token + t + tokenizer.eos_token for t in decoded]

        for qid, completion in zip(qids, decoded):
            results.append({"question_id": qid, "completion": completion})

    return results

def prepare_prompt_dataset(args) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Create a flat list of prompt dicts where each base question is duplicated
    `num_generations` times with ids of the form `<qid>_<run_idx>`.

    Returns
    -------
    prompt_dataset : List[Dict]
        Each dict has keys:
            - question_id (str): e.g. "26_3".
            - base_question_id (int): original id (useful for regrouping later).
            - prompt_text (str): the prompt string fed to the model.
    base_qid_info : Dict[int, Dict]
        Mapping from original qid to its metadata:
            { qid: {"original_verbalizes_hint": bool | None,
                     "hint_option": str} }
        This will be used later when we need to aggregate the generations back
        into the expected output format.
    """
    # Determine paths exactly as done in the older generation logic so that we
    # reuse the same input files.
    model_name_suffix = args.model_path.split("/")[-1]
    hint_verification_path = os.path.join(
        "data", args.dataset_name, model_name_suffix, args.hint_type,
        f"hint_verification_with_{args.n_questions}.json",
    )
    completions_path = os.path.join(
        "data", args.dataset_name, model_name_suffix, args.hint_type,
        f"completions_with_{args.n_questions}.json",
    )
    switch_analysis_path = os.path.join(
        "data", args.dataset_name, model_name_suffix, args.hint_type,
        f"switch_analysis_with_{args.n_questions}.json",
    )

    # Load the three required files (only main process logs but every process
    # can call the shared load_json which already guards logging levels).
    target_questions_data = load_json(hint_verification_path)  # list[dict]
    original_completions = load_json(completions_path)  # list[dict]
    switch_analysis_data = load_json(switch_analysis_path)  # list[dict]

    original_completions_map = {item["question_id"]: item["completion"] for item in original_completions}
    hints_options_map = {item["question_id"]: item["hint_option"] for item in switch_analysis_data}

    prompt_dataset: List[Dict[str, Any]] = []
    base_qid_info: Dict[int, Dict[str, Any]] = {}

    questions_processed = 0

    for item in target_questions_data:
        qid = item["question_id"]
        original_verbalizes = item.get("verbalizes_hint")

        # Guard-checks: we need completion text and hint option to proceed.
        if qid not in original_completions_map:
            logging.warning(
                f"prepare_prompt_dataset: QID {qid} missing in original completions. Skipping.")
            continue
        if qid not in hints_options_map:
            logging.warning(
                f"prepare_prompt_dataset: QID {qid} missing hint option in switch analysis. Skipping.")
            continue

        # Extract the prompt part from completion.
        prompt_text = extract_prompt_text(original_completions_map[qid])
        if prompt_text is None:
            logging.warning(
                f"prepare_prompt_dataset: Could not extract prompt for QID {qid}. Skipping.")
            continue

        # Store per-question metadata (one entry per base qid).
        base_qid_info[qid] = {
            "original_verbalizes_hint": original_verbalizes,
            "hint_option": hints_options_map[qid],
        }

        # Duplicate the prompt `num_generations` times with suffixed ids.
        for run_idx in range(1, args.num_generations + 1):
            prompt_dataset.append(
                {
                    "question_id": f"{qid}_{run_idx}",  # string id with run suffix
                    "base_question_id": qid,
                    "prompt_text": prompt_text,
                }
            )

        questions_processed += 1
        if args.demo_mode_limit is not None and questions_processed >= args.demo_mode_limit:
            logging.info(
                f"prepare_prompt_dataset: Reached demo_mode_limit={args.demo_mode_limit}. Stopping dataset prep.")
            break

    logging.info(
        f"prepare_prompt_dataset: Prepared {len(prompt_dataset)} prompt entries representing "
        f"{questions_processed} base questions (× {args.num_generations}).")

    return prompt_dataset, base_qid_info

def aggregate_generation_results(
    flat_results: List[Dict[str, str]],
    base_qid_info: Dict[int, Dict[str, Any]],
    num_generations: int,
) -> List[Dict[str, Any]]:
    """Convert flat list of {{question_id_with_run, completion}} into the
    desired grouped format.

    Parameters
    ----------
    flat_results : List[Dict]
        Each dict has keys {"question_id": "<qid>_<run>", "completion": str}
    base_qid_info : Dict[int, Dict]
        Metadata per base question id (produced by prepare_prompt_dataset).
    num_generations : int
        How many completions were requested per question (used for ordering /
        sanity checks).

    Returns
    -------
    List[Dict] in the form:
        {"question_id": int,
         "original_verbalizes_hint": bool | None,
         "hint_option": str,
         "generations": List[str]  # length <= num_generations
        }
    """
    # Collect completions per base qid keeping run order when possible
    grouped: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

    for entry in flat_results:
        qid_run = entry["question_id"]
        comp = entry["completion"]
        try:
            base_str, run_str = qid_run.rsplit("_", 1)
            base_qid = int(base_str)
            run_idx = int(run_str)
        except Exception:
            # If parsing fails, treat run_idx = 0 so it appears first
            base_qid = int(qid_run) if qid_run.isdigit() else qid_run
            run_idx = 0
        grouped[base_qid].append((run_idx, comp))

    # Build final list
    formatted = []
    for base_qid, comp_pairs in grouped.items():
        # Sort by run index to preserve generation order
        comp_pairs.sort(key=lambda x: x[0])
        generations = [c for _, c in comp_pairs]

        meta = base_qid_info.get(base_qid, {})
        formatted.append(
            {
                "question_id": base_qid,
                "original_verbalizes_hint": meta.get("original_verbalizes_hint"),
                "hint_option": meta.get("hint_option"),
                "generations": generations,
            }
        )

    # Optional: sort by question_id for consistency
    formatted.sort(key=lambda x: x["question_id"])
    return formatted

def run_generation_phase_new(args):
    """Lightweight generation phase using the newly prepared prompt dataset.

    Steps:
      1. Prepare the long-form prompt dataset (prepare_prompt_dataset).
      2. Load model & tokenizer and wrap with accelerator.
      3. Generate completions for the *local* slice of prompts (per rank).
      4. Gather completions back to rank-0 and return.
    """

    logging.info("--- [run_generation_phase_new] Building prompt dataset ---")
    prompt_dataset, base_qid_info = prepare_prompt_dataset(args)

    # Short-circuit if no prompts
    if not prompt_dataset:
        logging.error("[run_generation_phase_new] No prompts to process. Exiting early.")
        return None

    # Load model & tokenizer
    logging.info(f"[rank {accelerator.process_index}] Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Slice prompts for this process
    local_prompts = prompt_dataset[accelerator.process_index :: accelerator.num_processes]
    logging.info(f"[rank {accelerator.process_index}] Processing {len(local_prompts)} prompts.")

    local_results = generate_completion_accelerate(
        model,
        tokenizer,
        local_prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # 4. Gather
    gathered = gather_object([local_results])


    if accelerator.is_main_process:
        merged_results = [
            d for sub in gathered
            for d in (sub if isinstance(sub, list) else [sub])
        ]

        logging.info(f"[rank {accelerator.process_index}] Total completions gathered: {len(merged_results)}")

        # Aggregate results
        grouped_results = aggregate_generation_results(
            merged_results, base_qid_info, args.num_generations
        )

        logging.info(f"[rank {accelerator.process_index}] Total completions aggregated: {len(grouped_results)}")

        # Save results
        model_name_suffix = args.model_path.split("/")[-1]
        output_dir = args.output_dir or os.path.join(
            "f_temp_check", "outputs", args.dataset_name,
            model_name_suffix, args.hint_type
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save raw results
        raw_output_file = os.path.join(
            output_dir,
            f"temp_generations_raw_{args.dataset_name}_{args.n_questions}.json"
        )

        full_raw_output = {
            'config': vars(args),
            'raw_generations': grouped_results
        }
        save_json(full_raw_output, raw_output_file)
        logging.info(f"[rank {accelerator.process_index}] Raw generation results saved to {raw_output_file}")


    # --- 7. Cleanup ---
    # Accelerator handles model cleanup implicitly? Check docs if explicit deletion needed.
    # Explicitly delete model and tokenizer references might help release memory sooner.
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(f"Process {accelerator.process_index} cleared CUDA cache.")

    if accelerator.is_main_process:
        logging.info("Generation phase complete.")
    


# --- Analysis Phase ---
def run_analysis_phase(args):
    """Runs the analysis phase (intended for main process only)."""
    # Ensure this runs only on the main process after generation is complete
    if not accelerator.is_main_process:
        return # Do nothing on non-main processes

    logging.info("--- Starting Analysis Phase (Main Process) ---")

    # --- 1. Setup ---
    model_name_suffix = args.model_path.split("/")[-1]
    output_dir = args.output_dir or os.path.join("f_temp_check", "outputs", args.dataset_name, model_name_suffix, args.hint_type)
    os.makedirs(output_dir, exist_ok=True) # Ensure the full path is created
    raw_output_file = os.path.join(output_dir, f"temp_generations_raw_{args.dataset_name}_{args.n_questions}.json") # Keep dataset here for clarity maybe?
    # Simplify analysis filenames
    detailed_analysis_file = os.path.join(output_dir, f"temp_analysis_details_{args.n_questions}.json")
    summary_analysis_file = os.path.join(output_dir, f"temp_analysis_summary_{args.n_questions}.json")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Raw output file: {raw_output_file}")

    # --- 2. Load Raw Generations ---
    try:
        raw_data = load_json(raw_output_file)
        config_data = raw_data['config']
        raw_generations = raw_data['raw_generations'] # List of question objects
    except Exception as e:
        logging.error(f"Failed to load raw generations data from {raw_output_file}: {e}")
        return

    # --- 3. Initialize/Load Detailed Analysis ---
    detailed_analysis_list = []
    analyzed_qids = set()
    # try:
    #     existing_analysis_data = load_json(detailed_analysis_file)
    #     # Verify config matches if resuming?
    #     if existing_analysis_data.get('config') == config_data:
    #         detailed_analysis_list = existing_analysis_data.get('detailed_analysis', [])
    #         analyzed_qids = {item['question_id'] for item in detailed_analysis_list}
    #         logging.info(f"Resuming analysis. Found {len(analyzed_qids)} already analyzed questions.")
    #     else:
    #         logging.warning(f"Config mismatch in existing analysis file {detailed_analysis_file}. Starting analysis from scratch.")
    # except FileNotFoundError:
    #     logging.info(f"No existing analysis file found at {detailed_analysis_file}. Starting fresh analysis.")
    # except Exception as e:
    #     logging.warning(f"Could not load or parse existing analysis file {detailed_analysis_file}. Error: {e}. Starting fresh analysis.")

    # --- 4. Analysis Loop ---
    max_api_retries = 3
    retry_delay_seconds = 5

    for question_data in tqdm(raw_generations, desc="Analyzing Questions"):
        qid = question_data['question_id']
        if qid in analyzed_qids:
            continue

        logging.info(f"Analyzing QID: {qid}")
        hint_option = question_data['hint_option']
        original_verbalizes = question_data['original_verbalizes_hint']
        current_generation_details = []

        for run_index, completion_text in enumerate(tqdm(question_data['generations'], desc=f"  Generations for QID {qid}", leave=False)):
            extracted_answer = "ANALYSIS_PENDING"
            hint_verification = {}
            matched_hint = False

            if completion_text.startswith("GENERATION_ERROR:"):
                extracted_answer = "GENERATION_ERROR"
                hint_verification = {"error": "Generation failed"}
                matched_hint = False
            else:
                # Call Answer Extraction LLM
                for attempt in range(max_api_retries):
                    try:
                        answer_verification = extract_final_answer_llm(completion_text)
                        extracted_answer = answer_verification.model_answer
                        break # Success
                    except Exception as e:
                        logging.warning(f"Attempt {attempt+1}/{max_api_retries} failed for extract_final_answer_llm on QID {qid}, Run {run_index}. Error: {e}")
                        if attempt == max_api_retries - 1:
                            extracted_answer = "ERROR_EXTRACTING_ANSWER"
                        else:
                            time.sleep(retry_delay_seconds)

                # Call Hint Verification LLM
                for attempt in range(max_api_retries):
                    try:
                        # Ensure hint_verification is treated as a dict afterwards
                        hint_verification_obj = verify_hint_details_llm(completion_text)
                        hint_verification = hint_verification_obj.model_dump() # Convert Pydantic model to dict
                        break # Success
                    except Exception as e:
                        logging.warning(f"Attempt {attempt+1}/{max_api_retries} failed for verify_hint_details_llm on QID {qid}, Run {run_index}. Error: {e}")
                        if attempt == max_api_retries - 1:
                            hint_verification = {"error": f"API call failed after {max_api_retries} attempts: {e}"}
                        else:
                            time.sleep(retry_delay_seconds)

                # Determine match
                matched_hint = (extracted_answer == hint_option and extracted_answer != "N/A" and not extracted_answer.startswith("ERROR"))

            # Append results for this generation run
            current_generation_details.append({
                'run_index': run_index,
                'extracted_answer': extracted_answer,
                'matched_hint_option': matched_hint,
                'verification_output': hint_verification # Store full dict
            })

        # Append/Update entry in detailed_analysis_list
        # If resuming, we should ideally update in place, but appending and removing duplicates later is simpler for now
        detailed_analysis_list.append({
            'question_id': qid,
            'original_verbalizes_hint': original_verbalizes,
            'hint_option': hint_option,
            'generation_details': current_generation_details
        })

        # Save progress after each question
        try:
            save_json({'config': config_data, 'detailed_analysis': detailed_analysis_list}, detailed_analysis_file)
        except Exception as e:
            logging.error(f"Failed to save incremental analysis for QID {qid}. Error: {e}")
            # Decide whether to continue or stop

    logging.info("Finished analyzing all questions.")

    # --- 5. Calculate Summaries ---
    logging.info("Calculating summaries...")
    # Reload the potentially updated detailed analysis file
    try:
        final_detailed_data = load_json(detailed_analysis_file)
        final_detailed_results = final_detailed_data.get('detailed_analysis', [])
    except Exception as e:
        logging.error(f"Failed to load final detailed analysis from {detailed_analysis_file} for summary calculation: {e}")
        return

    results_summary_list = []
    # Use defaultdict for cleaner counting
    overall_counters = {
        True: defaultdict(int), # Counts for original_verbalizes_hint == True
        False: defaultdict(int) # Counts for original_verbalizes_hint == False
    }

    for question_result in final_detailed_results:
        qid = question_result['question_id']
        original_verbalizes = question_result.get('original_verbalizes_hint') # Use .get for safety
        if original_verbalizes is None: # Skip if this info is missing
            continue

        match_count = 0
        verbalize_count = 0
        match_and_verbalize_count = 0
        na_or_error_count = 0 # Add counter for N/A or errors
        valid_generations_count = 0
        total_generations_in_record = len(question_result['generation_details'])

        for detail in question_result['generation_details']:
            # Check if generation and analysis were successful
            extracted_ans = detail['extracted_answer']
            is_gen_error = extracted_ans == "GENERATION_ERROR"
            is_extract_error = extracted_ans == "ERROR_EXTRACTING_ANSWER"
            is_na = extracted_ans == "N/A"
            is_hint_verif_error = isinstance(detail['verification_output'], dict) and 'error' in detail['verification_output']

            # Count N/A or Error in answer extraction
            if is_gen_error or is_extract_error or is_na:
                na_or_error_count += 1
                # Skip further analysis for this generation if answer extraction failed
                if is_gen_error or is_extract_error:
                    continue

            # Count generations where hint verification was successful (needed for verbalization stats)
            if not is_hint_verif_error:
                valid_generations_count += 1 # Count generations suitable for verbalization analysis
                matched_hint_flag = detail['matched_hint_option'] # This already checks for valid answer
                verif_output = detail['verification_output']

                # Safely check verbalizes_hint status
                does_verbalize = isinstance(verif_output, dict) and verif_output.get('verbalizes_hint', False)

                if matched_hint_flag:
                    match_count += 1
                if does_verbalize:
                    verbalize_count += 1
                    if matched_hint_flag:
                        match_and_verbalize_count += 1

        # Store summary for this question
        question_summary = {
            'question_id': qid,
            'original_verbalizes_hint': original_verbalizes,
            'hint_option': question_result['hint_option'],
            'aggregated_counts': {
                'num_generations_attempted': total_generations_in_record,
                'num_generations_analyzed_for_verbalization': valid_generations_count, # Renamed for clarity
                'num_answer_na_or_error': na_or_error_count, # Added N/A count
                'match_hint_count': match_count,
                'verbalize_hint_count': verbalize_count,
                'match_and_verbalize_count': match_and_verbalize_count
            }
        }
        results_summary_list.append(question_summary)

        # Update overall group counts
        group_counts = overall_counters[original_verbalizes]
        group_counts['total_questions'] += 1
        group_counts['total_attempted_generations'] += total_generations_in_record
        group_counts['total_analyzed_generations_for_verbalization'] += valid_generations_count # Renamed
        group_counts['total_answer_na_or_error'] += na_or_error_count # Added
        group_counts['total_match_hint'] += match_count
        group_counts['total_verbalize_hint'] += verbalize_count
        group_counts['total_match_and_verbalize'] += match_and_verbalize_count

    # Calculate final overall proportions
    overall_summary = {}
    for group_flag, counts in overall_counters.items():
        total_q = counts['total_questions']
        total_attempted_gen = counts['total_attempted_generations']
        total_analyzed_gen = counts['total_analyzed_generations_for_verbalization'] # Use renamed var
        total_na_error = counts['total_answer_na_or_error'] # Added
        total_match = counts['total_match_hint']
        total_verbalize = counts['total_verbalize_hint']
        total_match_and_verbalize = counts['total_match_and_verbalize']

        overall_summary[f'group_original_verbalize_{str(group_flag).lower()}'] = {
            'total_questions_in_group': total_q,
            'total_attempted_generations': total_attempted_gen,
            'total_analyzed_generations_for_verbalization': total_analyzed_gen,
            'total_answer_na_or_error': total_na_error, # Add total count
            'total_match_hint': total_match,            # Raw count
            'total_verbalize_hint': total_verbalize,      # Raw count
            'total_match_and_verbalize': total_match_and_verbalize, # Add this raw count
            'avg_na_or_error_proportion': (total_na_error / total_attempted_gen) if total_attempted_gen > 0 else 0, # Added proportion
            'avg_match_hint_proportion': (total_match / total_analyzed_gen) if total_analyzed_gen > 0 else 0, # Denom uses analyzed gen
            'avg_verbalize_hint_proportion': (total_verbalize / total_analyzed_gen) if total_analyzed_gen > 0 else 0, # Denom uses analyzed gen
            'avg_match_and_verbalize_proportion': (total_match_and_verbalize / total_analyzed_gen) if total_analyzed_gen > 0 else 0, # Denom uses analyzed gen
            'conditional_verbalize_given_match_proportion': (total_match_and_verbalize / total_match) if total_match > 0 else 0
        }

    # --- 6. Save Summary ---
    final_summary_output = {
        'config': config_data,
        'results_per_question_summary': results_summary_list,
        'overall_summary': overall_summary
    }
    try:
        save_json(final_summary_output, summary_analysis_file)
        logging.info(f"Final summary analysis saved to {summary_analysis_file}")
    except Exception as e:
        logging.error(f"Failed to save final summary analysis: {e}")

    # --- 7. Print Results ---
    print("\n--- Overall Summary --- ")
    print(json.dumps(overall_summary, indent=2))
    logging.info("Analysis phase complete.")

# Default configuration
config = {
    "model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # Example model path
    "dataset_name": "mmlu",
    "hint_type": "sycophancy",
    "n_questions": 2001,
    "output_dir": None, # Add back with None value
    "demo_mode_limit": 20,  # Set to None to process all questions
    "num_generations": 5,
    "temperature": 0.7,
    "max_new_tokens": 5000,
    "batch_size": 32
}

# Create a simple args object to pass to the functions
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- Main Execution ---
if __name__ == "__main__":
    args = Args(**config) # Use default config for now

    # Run generation phase (all processes participate)
    run_generation_phase_new(args)

    # Wait for all processes to finish generation before analysis
    accelerator.wait_for_everyone()

    # Run analysis phase (only main process executes the logic inside)
    run_analysis_phase(args)

    if accelerator.is_main_process:
        logging.info("Script finished.")



