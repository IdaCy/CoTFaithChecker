import torch
import os
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
            torch_dtype=torch.bfloat16,
            output_hidden_states=True
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
    max_new_tokens: Optional[int] = 512, # Allow None
    output_location: str = "h_hidden_states/outputs",
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
    torch.manual_seed(0)
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

        encodings = tokenize_instructions(
            tokenizer,
            batch_prompt_texts, 
            chat_template
        )
        
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        #  capture resid_post activations 
        with torch.no_grad():
            prefix_out = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # ---- robust boundary search (multi-token safe) ---------------------------
        boundary_ids = tokenizer.encode("</think>", add_special_tokens=False)  # e.g. 3 ids

        def first_after(seq: torch.Tensor, pat: list[int]) -> int:
            """index just after first occurrence of pat in seq; else len(seq)."""
            pat = torch.tensor(pat, device=seq.device)
            for j in range(seq.size(0) - pat.size(0) + 1):
                if torch.equal(seq[j : j + pat.size(0)], pat):
                    return j + pat.size(0)        # inclusive; use j to cut before tag
            return seq.size(0)                    # not found → keep whole prompt

        prefix_end = [first_after(seq, boundary_ids) for seq in input_ids]
        # --------------------------------------------------------------------------

        layers = list(range(4, 33, 4))        # 4, 8, 12 … 32
        acts   = {}                            # will hold FP16 tensors

        for L in layers:
            h = prefix_out.hidden_states[L]    # (B, S, d)
            acts[f"layer_{L}"] = [
                h[b, :prefix_end[b]].to(torch.float16).cpu()
                for b in range(h.size(0))
            ]

        os.makedirs(output_location, exist_ok=True)
        torch.save(acts, f"{output_location}_batch_{i}.pt")
        #-----

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_tokens,
                do_sample=False,         # greedy ⇒ temperature ignored but leave it explicit
                temperature=0.0,
                top_p=1.0
            )

        #completions = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # decode only the newly generated tokens
        prompt_len = input_ids.size(1)            # length of the prompt portion
        gen_ids    = outputs[:, prompt_len:]      # slice off the prompt
        completions = tokenizer.batch_decode(     # ids → strings, clean
            gen_ids,
            skip_special_tokens=True              # drops <|begin_of_text|>, etc.
        )

        # store result
        for qid, completion_text in zip(batch_question_ids, completions):
            results.append({
                "question_id": qid,
                "completion": completion_text.lstrip()   # optional whitespace trim
            })

    return results
