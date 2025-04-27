import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Note: Type hint Int[Tensor, 'batch_size seq_len'] is not standard Python.
# Using torch.Tensor as a placeholder.
from typing import Dict, List, Tuple, Optional # Added Optional
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to get the appropriate device
def get_device():
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU.")
        return torch.device("cuda")
    # Mps backend can be problematic and need specific troubleshooting depending on the model/setup
    # elif torch.backends.mps.is_available():
    #     logging.info("MPS is available. Using Apple Silicon GPU.")
    #     return torch.device("mps")
    else:
        logging.info("CUDA/MPS not available. Using CPU.")
        return torch.device("cpu")

def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Loads the Hugging Face model and tokenizer onto the appropriate device.

    Args:
        model_name: The name or path of the Hugging Face model to use.
    Returns:
        A tuple containing the loaded model, tokenizer, and the device.
    Raises:
        RuntimeError: If model or tokenizer loading fails.
    """
    device = get_device()
    
    logging.info(f"Loading model and tokenizer: {model_path} onto {device}")
    try:
        model_name = model_path.split("/")[-1]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        model.eval().cuda() # Explicitly move model to the determined device
        model.padding_side='left'
        tokenizer.padding_side='left'
        
        if tokenizer.pad_token is None:
            logging.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer, model_name, device

    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise RuntimeError(f"Failed to load model/tokenizer: {model_path}") from e

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    chat_template: str
) -> torch.Tensor:
    """Tokenize instructions using the specified chat template."""
    # Ensure the template has a placeholder for the instruction
    if "{instruction}" not in chat_template:
        raise ValueError("Chat template must contain '{instruction}' placeholder.")
        
    prompts = [chat_template.format(instruction=instruction) for instruction in instructions]
    # Return only input_ids as per the original function's signature intent
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")

def generate_completion(
    model: AutoModelForCausalLM,      
    tokenizer: AutoTokenizer,    
    device: torch.device,        
    prompts: List[Dict], 
    chat_template: str,
    batch_size: int = 8, 
    max_new_tokens: Optional[int] = 512 # Allow None
    ) -> List[Dict]:
    """
    Generates completions for a list of prompts in batches using a pre-loaded model/tokenizer.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        device: The torch device the model is on.
        prompts: A list of dictionaries, containing 'question_id' and 'prompt_text'.
        chat_template: The chat template string (must contain '{instruction}').
        batch_size: The number of prompts to process in each batch.
        max_new_tokens: Maximum number of new tokens to generate. If None, uses a large default (e.g., 2048).

    Returns:
        A list of dictionaries, each containing 'question_id' and 'completion'.
    """
    # Model and tokenizer are already loaded and on the correct device
    results = []
    # model.eval() # Set model to evaluation mode

    # Handle max_new_tokens=None case
    gen_max_tokens = max_new_tokens if max_new_tokens is not None else 2048 # Default large value
    logging.info(f"Using max_new_tokens: {gen_max_tokens}")

    for i in range(0, len(prompts), batch_size):
        batch_prompts_data = prompts[i:i + batch_size]
        batch_prompt_texts = [item['prompt_text'] for item in batch_prompts_data]
        batch_question_ids = [item['question_id'] for item in batch_prompts_data]
        
        current_batch_size = len(batch_prompt_texts)
        logging.info(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size} (Size: {current_batch_size}, QIDs: {min(batch_question_ids)}-{max(batch_question_ids)})")

        # Format prompts using the chat template *before* tokenization
        formatted_prompts = [chat_template.format(instruction=text) for text in batch_prompt_texts]

        encodings = tokenize_instructions(
            tokenizer,
            batch_prompt_texts, # Pass original texts here, formatting happens inside tokenize_instructions now
            chat_template
        )
        
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)
        input_length = input_ids.shape[1] # Get length of input tokens

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_tokens, # Use the determined value
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id # Ensure pad token is set for generation
            )

        # Decode only the generated part for each item in the batch
        generated_texts = []
        # Get the string representation of the EOS token used for padding
        eos_token_str = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)

        for j in range(outputs.shape[0]):
            generated_ids = outputs[j, input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Remove any trailing EOS tokens (which are padding in this case)
            while generated_text.endswith(eos_token_str):
                generated_text = generated_text[:-len(eos_token_str)]

            generated_texts.append(generated_text)

        # Reconstruct the full completion by prepending the original formatted prompt
        # Note: We need the formatted prompts from *before* tokenization
        formatted_prompts_for_batch = [chat_template.format(instruction=item['prompt_text']) for item in batch_prompts_data]
        
        # Store results for the batch
        for qid, original_formatted_prompt, generated_part in zip(batch_question_ids, formatted_prompts_for_batch, generated_texts):
            full_completion_text = original_formatted_prompt + "<think>\n" + generated_part
            results.append({
                "question_id": qid,
                "completion": full_completion_text 
            })

    return results

# Remove cleanup code from here as it's now managed by the caller
# del model
# del tokenizer
# if device.type == 'cuda':
#     torch.cuda.empty_cache() 