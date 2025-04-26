
"Activation patching (proof-of-concept)."
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

__all__ = ["ActivationPatcher"]

class ActivationPatcher:
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_hidden(self, text: str, layer: int = -1):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.hidden_states[layer]

    def patch(self, source_prompt: str, donor_text: str, num_tokens: int = 16, layer_idx: int = -1):
        donor_hidden = self.get_hidden(donor_text, layer_idx)

        def _hook(module, inp, outp):
            outp[:, :num_tokens, :] = donor_hidden[:, :num_tokens, :]
            return outp

        handle = self.model.transformer.h[layer_idx].register_forward_hook(_hook)
        inputs = self.tokenizer(source_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        handle.remove()
        return logits
