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
from collections import defaultdict # Add defaultdict import
from accelerate import Accelerator # Added for multi-GPU
from accelerate.utils import gather_object # Corrected import location


# --- Project Specific Imports ---
# Assuming PYTHONPATH is set correctly or script is run from workspace root
try:
    from a_confirm_posthoc.utils.model_handler import load_model_and_tokenizer
except ImportError:
    print(os.getcwd())
    logging.error("Failed to import from a_confirm_posthoc.utils.model_handler. Ensure PYTHONPATH is set or run from project root.")
    exit(1)

# Import verification functions with aliases
try:
    from a_confirm_posthoc.eval.llm_verificator import verify_completion as extract_final_answer_llm
    from a_confirm_posthoc.eval.llm_verificator import Verification as AnswerVerificationResult
    from a_confirm_posthoc.eval.llm_hint_verificator import verify_completion as verify_hint_details_llm
    from a_confirm_posthoc.eval.llm_hint_verificator import Verification as HintVerificationResult
    from a_confirm_posthoc.eval.llm_hint_verificator import split_completion # If needed directly
except ImportError:
    logging.error("Failed to import verification functions from a_confirm_posthoc.eval. Ensure PYTHONPATH is set.")
    exit(1)

# --- Logging Setup ---
# Configure logging level based on main process
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Initialize accelerator early to control logging
accelerator = Accelerator()
logging.basicConfig(
    level=logging.INFO if accelerator.is_main_process else logging.ERROR, # Log INFO only on main process
    format='%(asctime)s - %(levelname)s - [Process %(process)d] - %(message)s'
)

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
    accelerator: Accelerator, # Use accelerator device
    num_generations: int,
    temperature: float,
    max_new_tokens: int,
    batch_size: int
) -> Dict[int, List[str]]:
    """
    Generates N completions for each prompt using batching and Accelerate.
    Returns a dictionary {question_id: [list_of_N_completions]} containing results for the current process.
    """
    # Each process generates for all prompts, results will be gathered later
    process_results_store = {item['question_id']: [] for item in prompts_data}
    prompt_map = {item['question_id']: item['prompt_text'] for item in prompts_data}
    question_ids_ordered = [item['question_id'] for item in prompts_data]

    logging.info(f"Process {accelerator.process_index} starting generation of {num_generations} completions for {len(prompts_data)} questions...")

    # The model is already prepared by accelerator, use its device
    device = accelerator.device

    for n in range(num_generations):
        logging.info(f"--- Process {accelerator.process_index} Generating run {n+1}/{num_generations} ---")
        # Process in batches
        # Use tqdm only on the main process to avoid multiple progress bars
        iterable = range(0, len(question_ids_ordered), batch_size)
        if accelerator.is_main_process:
            iterable = tqdm(iterable, desc=f"Batch for Run {n+1}")

        for i in iterable:
            batch_qids = question_ids_ordered[i:i + batch_size]
            batch_prompts = [prompt_map[qid] for qid in batch_qids]

            # Tokenize the batch
            encodings = tokenizer(batch_prompts,
            padding=True,
            truncation=False, # Ensure truncation is False or handled carefully
            return_tensors="pt")#.to(device) # Move tensors to the correct device

            # Manually move to device - safer with accelerate? Check docs. Usually prepare handles model device.
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)


            # Generate
            try:
                with torch.no_grad():
                    input_length = input_ids.shape[1] # Get input length BEFORE generation
                    # Accelerator handles model placement, generate should work
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=tokenizer.eos_token_id # Crucial for open-ended generation with padding
                        # top_p=0.95 # Added common sampling parameter
                    )

                # Decode only the generated part and reconstruct
                # Ensure decoding happens correctly potentially off CPU if needed after generation
                # outputs might be on GPU, move to CPU for decoding if tokenizer expects CPU tensors
                decoded_outputs = outputs.cpu() # Move generated IDs to CPU for decoding

                for j, qid in enumerate(batch_qids):
                    original_prompt = batch_prompts[j]
                    # Use the CPU tensor for slicing and decoding
                    generated_ids = decoded_outputs[j, input_length:]
                    generated_part = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    # Construct the final string with <think> token
                    # Ensure original_prompt ends correctly (it should from extract_prompt_text)
                    final_completion = original_prompt + generated_part.strip() + tokenizer.eos_token

                    process_results_store[qid].append(final_completion)

            except Exception as e:
                 logging.error(f"Error during generation for batch starting with QID {batch_qids[0]} in run {n+1} on process {accelerator.process_index}: {e}")
                 # Store error marker or skip? For now, let's add error markers
                 for qid in batch_qids:
                    process_results_store[qid].append(f"GENERATION_ERROR: {e}")

            # Optional: Clear cache between batches if memory pressure is high
            # torch.cuda.empty_cache() # This might interfere with accelerate, use with caution

    logging.info(f"Process {accelerator.process_index} finished generation.")
    # Return results from this specific process
    return process_results_store


# --- Main Execution Logic ---
def run_generation_phase(args, accelerator): # Pass accelerator
    """Runs the generation phase using Accelerate."""
    # Use accelerator logging
    logging.info("--- Starting Generation Phase ---")

    # --- 1. Setup ---
    # Construct model name from path for output files
    model_name_suffix = args.model_path.split("/")[-1]
    # Output dir setup only on main process
    output_dir = None
    raw_output_file = None

    print(f"DEMO LIMIT: {args.demo_mode_limit}")

    if accelerator.is_main_process:
        output_dir = args.output_dir or os.path.join("f_temp_check", "outputs", args.dataset_name, model_name_suffix, args.hint_type)
        os.makedirs(output_dir, exist_ok=True)
        raw_output_file = os.path.join(output_dir, f"temp_generations_raw_{args.dataset_name}_{args.n_questions}.json")
        logging.info(f"Output directory (main process): {output_dir}")
        logging.info(f"Raw output file (main process): {raw_output_file}")

    # --- 2. Load Model (on all processes, Accelerate handles placement) ---
    model, tokenizer = None, None
    try:
        # Load model and tokenizer on CPU first to avoid OOM issues on rank 0?
        # Or let load_model_and_tokenizer handle device placement if aware
        # For simplicity, assume load_model_and_tokenizer loads onto CPU or specified device
        # We override device placement with accelerator later.
        # Temporarily load to CPU might be safest if load_model_and_tokenizer defaults to GPU 0
        # This part might need adjustment based on load_model_and_tokenizer's implementation
        # Assuming it returns model on CPU or first GPU, accelerator.prepare will handle it
        _model, _tokenizer, loaded_model_name, _ = load_model_and_tokenizer(args.model_path) # Modify load func if needed? Or just load standardly.

        # Prepare model and tokenizer with accelerator
        # This moves the model to the correct device(s)
        model = accelerator.prepare(_model)
        # Tokenizer usually doesn't need prepare unless it has state/buffers
        tokenizer = _tokenizer
        logging.info(f"Model and tokenizer prepared by Accelerate on process {accelerator.process_index}.")

    except Exception as e:
        logging.error(f"Failed to load/prepare model on process {accelerator.process_index}: {e}")
        # Use accelerator.set_trigger() or similar for early exit across processes if needed
        return # Exit generation phase for this process

    # --- 3. Load Input Data (Only main process needs to load initially, then broadcast/scatter?) ---
    # Simplest: Each process loads the data needed. Less efficient but easier.
    # Let's try loading on main process and broadcasting the necessary parts.
    prompts_for_generation = []
    target_info = {}
    if accelerator.is_main_process:
        try:
            hint_verification_path = os.path.join("data", args.dataset_name, model_name_suffix, args.hint_type, f"hint_verification_with_{args.n_questions}.json")
            target_questions_data = load_json(hint_verification_path) # List of dicts

            completions_path = os.path.join("data", args.dataset_name, model_name_suffix, args.hint_type, f"completions_with_{args.n_questions}.json")
            original_completions = load_json(completions_path) # List of dicts
            original_completions_map = {item['question_id']: item['completion'] for item in original_completions}

            switch_analysis_path = os.path.join("data", args.dataset_name, model_name_suffix, args.hint_type, f"switch_analysis_with_{args.n_questions}.json")
            switch_analysis_data = load_json(switch_analysis_path) # List of dicts
            hints_options_map = {item['question_id']: item['hint_option'] for item in switch_analysis_data}

            # --- 4. Prepare Prompts & Target Info (on main process) ---
            questions_processed_count = 0
            logging.info("Preparing prompts on main process...")
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

                prompts_for_generation.append({'question_id': qid, 'prompt_text': prompt_text})
                target_info[qid] = {'original_verbalizes_hint': original_verbalizes, 'hint_option': hint_option}

                questions_processed_count += 1
                if args.demo_mode_limit is not None and questions_processed_count >= args.demo_mode_limit:
                    logging.info(f"Reached demo mode limit of {args.demo_mode_limit} questions.")
                    break

            if not prompts_for_generation:
                logging.error("No valid prompts prepared for generation on main process. Exiting.")
                # Need mechanism to stop other processes
                # For now, return empty list/dict to signify issue
                prompts_for_generation = []
                target_info = {}

        except Exception as e:
            logging.error(f"Failed to load necessary input data files on main process: {e}")
            # Signal error state
            prompts_for_generation = None # Use None to signal error
            target_info = None

    # Broadcast the prepared data from main process to all others
    # Use object broadcasting
    data_to_broadcast = [prompts_for_generation, target_info]
    # Use accelerator.broadcast_object_list which is simpler for multiple objects
    broadcast_results = accelerator.broadcast_object_list(data_to_broadcast, from_process=0)
    # Unpack results on all processes
    prompts_for_generation, target_info = broadcast_results

    # Check for error signal
    if prompts_for_generation is None or target_info is None:
         logging.error(f"Process {accelerator.process_index} did not receive valid data. Exiting generation phase.")
         return

    if not prompts_for_generation:
         logging.warning(f"Process {accelerator.process_index} received empty prompts list. Skipping generation.")
         return # Or handle as appropriate

    logging.info(f"Process {accelerator.process_index} received {len(prompts_for_generation)} prompts for generation.")

    # --- 5. Generate Completions (on all processes) ---
    start_time = time.time()
    # Each process generates for *all* prompts_for_generation list
    # This is simpler than splitting data but means redundant computation if N_GPU > 1
    # but Accelerate might optimize model parallelism under the hood.
    # We need to gather results later.
    process_generated_data = generate_n_completions_batched(
        prompts_data=prompts_for_generation,
        model=model, # Use the prepared model
        tokenizer=tokenizer,
        accelerator=accelerator, # Pass accelerator
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    end_time = time.time()
    logging.info(f"Generation on process {accelerator.process_index} finished in {end_time - start_time:.2f} seconds.")

    # --- 6. Gather and Save Raw Results (on main process) ---
    # Gather results from all processes to the main process
    # gather_object collects Python objects from all processes into a list on the main process
    all_process_results = gather_object(process_generated_data)

    if accelerator.is_main_process:
        logging.info("Gathering results on main process...")
        # all_process_results is a list [dict_from_proc0, dict_from_proc1, ...]
        # We need to merge these. Since each process computed for *all* qids,
        # the results for a given qid will be duplicated across the list.
        # Let's assume for now each process generated the *same* set of results (non-deterministic sampling aside)
        # and just take the results from the first process (process 0).
        # If sampling leads to different results per process, a merging strategy is needed.
        # Simplest: Assume results are sufficiently similar or take the first one.
        if all_process_results:
            gathered_data = all_process_results[0] # Take results from the first process
        else:
            gathered_data = {} # Handle case where gathering might fail or return empty

        raw_results_list = []
        for qid, generations_list in gathered_data.items():
            if qid in target_info: # Ensure we only save results for prompts we prepared
                raw_results_list.append({
                    'question_id': qid,
                    'original_verbalizes_hint': target_info[qid]['original_verbalizes_hint'],
                    'hint_option': target_info[qid]['hint_option'],
                    'generations': generations_list
                })
            else:
                 logging.warning(f"Generated data for QID {qid} but it wasn't in the initial target info (main process). Ignoring.")

        config_data = vars(args) # Save command line args used

        full_raw_output = {'config': config_data, 'raw_generations': raw_results_list}
        if raw_output_file: # Ensure filename is available
            save_json(full_raw_output, raw_output_file)
            logging.info(f"Raw generation results saved to {raw_output_file} by main process.")
        else:
            logging.error("Output file path not set on main process. Cannot save raw results.")

    # --- 7. Cleanup ---
    logging.info(f"Cleaning up model and GPU memory on process {accelerator.process_index}...")
    del model
    del tokenizer
    # accelerator handles device cleanup generally, but explicit cache clearing might help
    if accelerator.state.device.type == 'cuda':
         torch.cuda.empty_cache()
    logging.info(f"Generation phase complete on process {accelerator.process_index}.")
    # Wait for all processes to finish before main process exits generation phase
    accelerator.wait_for_everyone()


def run_analysis_phase(args):
    """Runs the analysis phase."""
    logging.info("--- Starting Analysis Phase ---")

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
    "demo_mode_limit": 10,  # Set to None to process all questions
    "num_generations": 10,
    "temperature": 0.7,
    "max_new_tokens": 5000,
    "batch_size": 50
}

# Create a simple args object to pass to the functions
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

args = Args(**config)

# Initialize accelerator outside phases
# accelerator = Accelerator() # Moved initialization earlier

# --- Run Phases ---
# Generation runs on all processes managed by Accelerate
run_generation_phase(args, accelerator)

# Analysis runs only on the main process
if accelerator.is_main_process:
    logging.info("Running analysis phase on main process...")
    run_analysis_phase(args)
else:
    logging.info(f"Process {accelerator.process_index} skipping analysis phase.")

# Ensure all processes reach this point before exiting
accelerator.wait_for_everyone()
logging.info(f"Script finished on process {accelerator.process_index}.")
# Remove final duplicate logging message
# logging.info("Script finished.")
