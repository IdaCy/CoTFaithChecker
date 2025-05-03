import torch
import sys # Import sys
import json
import os
import logging
import argparse
from accelerate import Accelerator # Add Accelerator
from accelerate.utils import gather_object, broadcast_object_list # Add gather_object and broadcast_object_list
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

# accelerate launch f_temp_check/temp_robust_new_accelerate_long_df2_2.py


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
    logging.getLogger().setLevel(logging.WARNING) # Reduce logging level for non-main processes
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
    batch_size: int,
) -> Dict[int, List[Optional[str]]]:
    """
    Given a *flattened* list where each element contains

        {"question_id": int, "run_id": int, "prompt_text": str}

    this function generates **one** completion per element.  It returns a
    dictionary mapping each `question_id` to a list of length
    `num_generations`, with the completion inserted at `run_id`.
    """

    # Ensure we are on the correct device.
    device = accelerator.device

    gen_model = model.module if hasattr(model, "module") else model

    # Build a results structure pre-filled with None placeholders.
    unique_qids = {item["question_id"] for item in prompts_data}
    local_results_store: Dict[int, List[Optional[str]]] = {
        qid: [None] * num_generations for qid in unique_qids
    }

    num_entries = len(prompts_data)
    if num_entries == 0:
        logging.warning(
            f"Process {accelerator.process_index} received no prompt entries."
        )
        return {}

    logging.info(
        f"Process {accelerator.process_index}: Generating one completion for each of {num_entries} prompt entries …"
    )

    # Iterate once over the flattened list in batches
    iterable = range(0, num_entries, batch_size)
    if accelerator.is_main_process:
        iterable = tqdm(iterable, desc="Generation Batches (Main Process)")
    else:
        iterable = tqdm(
            iterable,
            desc=f"Generation Batches (Process {accelerator.process_index})",
            disable=not accelerator.is_local_main_process,
        )

    # Convenience view of the data to index quickly during the loop.
    # We keep it as a list because we only need slice access.
    prompts_data_list = prompts_data

    for start_idx in iterable:
        batch_items = prompts_data_list[start_idx : start_idx + batch_size]

        batch_prompts = [item["prompt_text"] for item in batch_items]

        encodings = tokenizer(
            batch_prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        try:
            with torch.no_grad():
                input_length = input_ids.shape[1]

                outputs = gen_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = outputs[:, input_length:]
            generated_parts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for row_idx, item in enumerate(batch_items):
                qid = item["question_id"]
                run_id = item["run_id"]
                prompt_text = item["prompt_text"]

                completion_text = (
                    prompt_text + generated_parts[row_idx].strip() + tokenizer.eos_token
                )

                local_results_store[qid][run_id] = completion_text

        except Exception as e:
            logging.error(
                f"Process {accelerator.process_index}: Error during generation starting with entry index {start_idx}: {e}"
            )
            for item in batch_items:
                qid = item["question_id"]
                run_id = item["run_id"]
                local_results_store[qid][run_id] = f"GENERATION_ERROR: {e}"

        # Optional: clear cache
        # torch.cuda.empty_cache()

    return local_results_store


# --- Main Execution Logic ---
def run_generation_phase(args):
    """Runs the generation phase using accelerate."""
    # Use accelerator.is_main_process for initial setup logs
    if accelerator.is_main_process:
        logging.info("--- Starting Generation Phase ---")

    # --- 1. Setup (Main Process Only for File Paths) ---
    model_name_suffix = args.model_path.split("/")[-1]
    output_dir = args.output_dir or os.path.join("f_temp_check", "outputs", args.dataset_name, model_name_suffix, args.hint_type)
    # Ensure directory exists (only main process needs to create it)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    raw_output_file = os.path.join(output_dir, f"temp_generations_raw_{args.dataset_name}_{args.n_questions}.json")
    if accelerator.is_main_process:
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Raw output file: {raw_output_file}")

    # --- 2. Load Model (Using Accelerate) ---
    device = accelerator.device # Get device from accelerator
    logging.info(f"Process {accelerator.process_index}: Loading model and tokenizer: {args.model_path} onto {device}")
    try:
        # Load model and tokenizer directly here, similar to parallelization/driver.py
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 # Or adjust as needed
            # Add other loading args like quantization if used previously
        )
        # Set padding side for consistency
        model.config.use_cache = False # Often recommended with gradient checkpointing or multi-GPU
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            logging.warning(f"Process {accelerator.process_index}: Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id # Ensure model config matches

        # Prepare model and tokenizer with accelerator
        model, tokenizer = accelerator.prepare(model, tokenizer)
        logging.info(f"Process {accelerator.process_index}: Model and tokenizer prepared.")

    except Exception as e:
        logging.error(f"Process {accelerator.process_index}: Failed to load or prepare model: {e}")
        # Ensure all processes exit if loading fails on any process
        accelerator.set_trigger() # Signal other processes to possibly exit gracefully
        return # Exit generation phase on this process

    # Check if any process triggered an exit
    if accelerator.check_trigger():
        logging.error(f"Process {accelerator.process_index}: Exiting due to trigger from another process (likely model loading failure).")
        return

    # --- 3. Load Input Data (all ranks) ---
    target_questions_data = []
    original_completions_map: Dict[int, str] = {}
    hints_options_map: Dict[int, str] = {}

    try:
        hint_verification_path = os.path.join(
            "data",
            args.dataset_name,
            model_name_suffix,
            args.hint_type,
            f"hint_verification_with_{args.n_questions}.json",
        )
        target_questions_data = load_json(hint_verification_path)

        completions_path = os.path.join(
            "data",
            args.dataset_name,
            model_name_suffix,
            args.hint_type,
            f"completions_with_{args.n_questions}.json",
        )
        original_completions = load_json(completions_path)
        original_completions_map = {
            item["question_id"]: item["completion"] for item in original_completions
        }

        switch_analysis_path = os.path.join(
            "data",
            args.dataset_name,
            model_name_suffix,
            args.hint_type,
            f"switch_analysis_with_{args.n_questions}.json",
        )
        switch_analysis_data = load_json(switch_analysis_path)
        hints_options_map = {
            item["question_id"]: item["hint_option"] for item in switch_analysis_data
        }

    except Exception as e:
        logging.error(
            f"Process {accelerator.process_index}: Failed to load necessary input data files: {e}"
        )
        accelerator.set_trigger()
        return

    # Check if main process triggered an exit during data loading
    if accelerator.check_trigger():
        logging.error(f"Process {accelerator.process_index}: Exiting due to trigger from main process (likely data loading failure).")
        return


    # --- 4. Prepare Prompts & Target Info (all ranks) ---
    all_prompts_for_generation: List[Dict] = []
    all_target_info: Dict[int, Dict] = {}

    logging.info(f"Process {accelerator.process_index}: Preparing prompts…")
    questions_processed_count = 0
    for item in target_questions_data:
        qid = item['question_id']
        original_verbalizes = item.get('verbalizes_hint', None)

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

        all_prompts_for_generation.append({
            'question_id': qid,
            'prompt_text': prompt_text
        })
        all_target_info[qid] = {
            'original_verbalizes_hint': original_verbalizes,
            'hint_option': hint_option
        }

        questions_processed_count += 1
        if args.demo_mode_limit is not None and questions_processed_count >= args.demo_mode_limit:
            logging.info(f"Reached demo mode limit of {args.demo_mode_limit} questions.")
            break

    if not all_prompts_for_generation:
        logging.error(
            f"Process {accelerator.process_index}: No valid prompts prepared. Triggering shutdown."
        )
        accelerator.set_trigger()
    else:
        logging.info(
            f"Process {accelerator.process_index}: Prepared {len(all_prompts_for_generation)} base prompts. Expanding with run_id…"
        )

        # Expand each prompt into (qid, run_id, prompt_text)
        expanded_prompt_list = []
        for prompt_entry in all_prompts_for_generation:
            for run_id in range(args.num_generations):
                expanded_prompt_list.append({
                    'question_id': prompt_entry['question_id'],
                    'run_id': run_id,
                    'prompt_text': prompt_entry['prompt_text']
                })

        logging.info(
            f"Process {accelerator.process_index}: Produced {len(expanded_prompt_list)} prompt entries after expansion."
        )
        # Replace the prompt list with the expanded version for broadcasting
        all_prompts_for_generation = expanded_prompt_list

    # Check trigger again after main process prompt preparation
    if accelerator.check_trigger():
        logging.warning(f"Process {accelerator.process_index}: Exiting - No prompts prepared by main process.")
        return

    # We keep `all_target_info` only on the main rank; other ranks don't need it
    final_target_info = all_target_info  # every rank has it now; only main uses it

    # ------------------------------------------------------------------
    # Simpler Accelerate pattern: Every rank sees the full prompt list
    # (constructed independently) and takes its slice with Python stride.
    # ------------------------------------------------------------------

    complete_prompt_list = all_prompts_for_generation
    rank, world = accelerator.process_index, accelerator.num_processes
    prompts_for_this_process = complete_prompt_list[rank::world]

    num_prompts_local = len(prompts_for_this_process)
    logging.info(
        f"Process {accelerator.process_index} will process {num_prompts_local} prompt entries."
    )


    # --- 5. Generate Completions (Distributed) ---
    start_time = time.time()
    # Each process generates completions for its subset of prompts
    local_generated_data = generate_n_completions_batched(
        prompts_data=prompts_for_this_process, # Pass the slice for this process
        model=model, # Pass the prepared model
        tokenizer=tokenizer, # Pass the prepared tokenizer
        device=device, # Pass the device
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size # This batch size is per-GPU
    )
    end_time = time.time()
    logging.info(f"Process {accelerator.process_index} finished generation in {end_time - start_time:.2f} seconds.")

    # --- 6. Gather and Structure Results (Main Process Only) ---
    # Gather results from all processes
    gathered_results = gather_object([local_generated_data])  # List of dicts, one from each process

    if accelerator.is_main_process:
        logging.info("Main process gathering and structuring results...")
        # Merge the dictionaries from all processes (some question IDs are split
        # across processes, so we need to combine their lists element-wise).
        final_generated_data: Dict[int, List[Optional[str]]] = {}

        for proc_result_dict in gathered_results:
            for qid, gen_list in proc_result_dict.items():
                if qid not in final_generated_data:
                    final_generated_data[qid] = gen_list
                else:
                    merged = final_generated_data[qid]
                    for idx, val in enumerate(gen_list):
                        if merged[idx] is None and val is not None:
                            merged[idx] = val
                    final_generated_data[qid] = merged

        # Structure the results using the broadcasted target_info
        raw_results_list = []
        processed_qids_count = 0
        for qid, generations_list in final_generated_data.items():
             # Use the combined final_target_info obtained earlier
            if qid in final_target_info:
                raw_results_list.append({
                    'question_id': qid,
                    'original_verbalizes_hint': final_target_info[qid]['original_verbalizes_hint'],
                    'hint_option': final_target_info[qid]['hint_option'],
                    'generations': generations_list
                })
                processed_qids_count += 1
            else:
                 logging.warning(f"Generated data for QID {qid} but it wasn't in the target info. Ignoring.")

        logging.info(f"Main process structured results for {processed_qids_count} questions.")

        # Save the combined results (only on main process)
        config_data = vars(args) # Save command line args used
        full_raw_output = {'config': config_data, 'raw_generations': raw_results_list}
        save_json(full_raw_output, raw_output_file) # save_json already checks for main process
        logging.info(f"Raw generation results saved to {raw_output_file}")

    # --- 7. Cleanup ---
    # Accelerator handles model cleanup implicitly? Check docs if explicit deletion needed.
    # Explicitly delete model and tokenizer references might help release memory sooner.
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    try:
        existing_analysis_data = load_json(detailed_analysis_file)
        # Verify config matches if resuming?
        if existing_analysis_data.get('config') == config_data:
            detailed_analysis_list = existing_analysis_data.get('detailed_analysis', [])
            analyzed_qids = {item['question_id'] for item in detailed_analysis_list}
            logging.info(f"Resuming analysis. Found {len(analyzed_qids)} already analyzed questions.")
        else:
            logging.warning(f"Config mismatch in existing analysis file {detailed_analysis_file}. Starting analysis from scratch.")
    except FileNotFoundError:
        logging.info(f"No existing analysis file found at {detailed_analysis_file}. Starting fresh analysis.")
    except Exception as e:
        logging.warning(f"Could not load or parse existing analysis file {detailed_analysis_file}. Error: {e}. Starting fresh analysis.")

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
    "demo_mode_limit": 30,  # Set to None to process all questions
    "num_generations": 10,
    "temperature": 0.7,
    "max_new_tokens": 5000,
    "batch_size": 10
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
    run_generation_phase(args)

    # Wait for all processes to finish generation before analysis
    accelerator.wait_for_everyone()

    # Run analysis phase (only main process executes the logic inside)
    run_analysis_phase(args)

    if accelerator.is_main_process:
        logging.info("Script finished.")
