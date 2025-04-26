import json
import os
from typing import List, Dict, Optional
import torch
import logging
import time # For timing

# Assuming utils are in the same directory or PYTHONPATH is set correctly
from a_confirm_posthoc.utils.prompt_constructor import construct_prompt
from a_confirm_posthoc.utils.model_handler import generate_completion, load_model_and_tokenizer

# Setup basic logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define known chat templates (can be expanded)
# Using the structure expected by tokenize_instructions: {instruction}
KNOWN_CHAT_TEMPLATES = {
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "qwen": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

}

def get_chat_template(model_name: str) -> str:
    """Selects a chat template based on the model name."""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower or "llama" in model_name_lower:
        return KNOWN_CHAT_TEMPLATES["llama3"]
    elif "qwen" in model_name_lower: # Assuming mistral instruct models
        return KNOWN_CHAT_TEMPLATES["qwen"]
    # Add more specific checks if needed
    else:
        logging.warning(f"No specific chat template found for {model_name}. Using default.")
        return KNOWN_CHAT_TEMPLATES["default"]

def load_data(data_path: str) -> List[Dict]:
    """Loads JSON data from a file, handling potential errors."""
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return [] # Return empty list to indicate failure
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {data_path}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {data_path}: {e}")
        return []

def save_results(results: List[Dict], dataset_name: str, hint_type: str, model_name:str, n_questions: int):
    """Saves the results to a JSON file in the dataset/model/hint_type directory."""
    # Construct path including model name directory and remove model name from filename
    output_path = os.path.join("data", dataset_name, model_name, hint_type, f"completions_with_{str(n_questions)}.json")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_path}: {e}")


def generate_dataset_completions(
    model,
    tokenizer,
    model_name,
    device,
    dataset_name: str,
    hint_types: List[str], # e.g., ["gsm_mc_urgency", "gsm_mc_psychophancy"]
    batch_size: int = 8,
    max_new_tokens: Optional[int] = 512,
    n_questions: Optional[int] = None,
):
    """
    Loads a model, processes datasets for specified hint types (with and without hints),
    generates completions, and saves the results.

    Args:
        model_name: Name/path of the Hugging Face model.
        hint_types: List of hint type identifiers (used to find data files like f'{data_base_dir}/{ht}.json').
        batch_size: Batch size for generation.
        max_new_tokens: Maximum number of new tokens. None means generate until EOS.
    """
    start_time = time.time()
    
        
    # --- 2. Select Chat Template --- 
    chat_template = get_chat_template(model_name)
    logging.info(f"Using chat template: {chat_template}")

    # --- 3. Process each hint type dataset --- 
    for hint_type in hint_types:
        logging.info(f"--- Processing dataset for hint type: {hint_type} ---")
        # Use the standardized input filename
        questions_data_path = os.path.join("data", dataset_name, "input_mcq_data.json")
        # Load hints from the dataset directory, using the new filename format
        hints_data_path = os.path.join("data", dataset_name, f"hints_{hint_type}.json")
        
        # Load questions and hints data
        data = load_data(questions_data_path)[:n_questions]  # Only use the first 10 entries
        hints = load_data(hints_data_path)[:n_questions]
        
        # Create a dictionary mapping question_id to the entire hint object
        # Handles cases where hints file might be empty or not found (load_data returns [])
        hint_data_dict = {hint["question_id"]: hint for hint in hints}
        
        # Add hint_text to each question entry
        for entry in data:
            # Retrieve the full hint object for the question ID
            hint_entry = hint_data_dict.get(entry["question_id"])
            # Add hint_text if a hint entry exists for this question ID
            entry["hint_text"] = hint_entry.get("hint_text") if hint_entry else None

            

        # --- Generate with hints --- 
        logging.info(f"Generating completions for {hint_type}...")
        prompts = []
        for entry in data:
            prompt_text = construct_prompt(entry) # Use original entry with hint
            prompts.append({"question_id": entry["question_id"], "prompt_text": prompt_text})
            
        
        results = generate_completion(
            model, tokenizer, device, prompts, 
            chat_template, batch_size, max_new_tokens
        )

        save_results(results, dataset_name, hint_type, model_name, n_questions)
        

    # --- 4. Cleanup --- 
    # logging.info("Cleaning up model and tokenizer...")
    # del model
    # del tokenizer
    # if device.type == 'cuda':
    #     torch.cuda.empty_cache()
    #     logging.info("CUDA cache cleared.")
        
    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")



# Example of how to call the function (replace main() block)
if __name__ == "__main__":
    # Example Usage: Load model/tokenizer first (implementation depends on model_handler.py)
    # model, tokenizer, model_name_loaded, device = load_model_and_tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # generate_dataset_completions(
    #     model=model,
    #     tokenizer=tokenizer,
    #     model_name=model_name_loaded,
    #     device=device,
    #     dataset_name="gsm8k",  # Provide the dataset name
    #     hint_types=["induced_urgency"],
    #     batch_size=4,
    #     max_new_tokens=1024,
    #     n_questions=10 # Example: limit to 10 questions
    # )
    pass # Placeholder, actual model loading needed


# Remove the old main() function and argparse code 
