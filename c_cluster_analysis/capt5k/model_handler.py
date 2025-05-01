import torch, logging
from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    StoppingCriteria, StoppingCriteriaList
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# helpers
def get_device():
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU.")
        return torch.device("cuda")
    logging.info("CUDA not available. Using CPU.")
    return torch.device("cpu")

def load_model_and_tokenizer(model_path: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    device = get_device()
    logging.info(f"Loading model and tokenizer: {model_path} onto {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.bfloat16)
    model.eval()
    model.padding_side = "left"
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer, model_path.split("/")[-1], device

def tokenize_instructions(tokenizer, instructions, chat_template):
    if "{instruction}" not in chat_template:
        raise ValueError("Chat template must contain '{instruction}'.")
    prompts = [chat_template.format(instruction=x) for x in instructions]
    return tokenizer(prompts, padding=True, truncation=False,
                     return_tensors="pt")

class StopOnThinkEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        tail = input_ids[0, -len(self.stop_ids):]
        return torch.equal(tail,
                           torch.tensor(self.stop_ids, device=input_ids.device))

# ------------------------------------------------------------
# generate_completion
# ------------------------------------------------------------
def generate_completion(
    model, tokenizer, device,
    prompts: List[Dict], chat_template: str,
    batch_size: int = 8, max_new_tokens: Optional[int] = 512
) -> List[Dict]:

    results, gen_max = [], max_new_tokens or 2048
    logging.info(f"Using max_new_tokens: {gen_max}")

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        texts = [p["prompt_text"] for p in batch]
        qids  = [p["question_id"]  for p in batch]

        logging.info(f"Processing batch {i//batch_size+1}/"
                     f"{(len(prompts)+batch_size-1)//batch_size} "
                     f"(size {len(texts)}, QIDs {min(qids)}-{max(qids)})")

        enc = tokenize_instructions(tokenizer, texts, chat_template)

        # unwrap DDP if present
        gen_model  = model.module if hasattr(model, "module") else model
        gen_device = next(gen_model.parameters()).device

        input_ids      = enc["input_ids"].to(gen_device)
        attention_mask = enc["attention_mask"].to(gen_device)
        inp_len        = input_ids.shape[1]

        with torch.no_grad():
            outputs = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                stopping_criteria=StoppingCriteriaList([StopOnThinkEnd(tokenizer)]),
                max_new_tokens=gen_max,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        eos_tok = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False)
        for qid, out in zip(qids, outputs):
            gen_ids  = out[inp_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            while gen_text.endswith(eos_tok):
                gen_text = gen_text[:-len(eos_tok)]

            gen_text = gen_text.lstrip()
            think_idx = gen_text.find("<think>")
            span = gen_text[think_idx:] if think_idx != -1 else gen_text
            if "</think>" in span:
                span = span.split("</think>")[0]

            results.append({"question_id": qid, "completion": span})

    return results
