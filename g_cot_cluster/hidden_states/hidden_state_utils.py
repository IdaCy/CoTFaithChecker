
"""Utilities for extracting and manipulating hidden states from LLMs."""

from __future__ import annotations
from functools import lru_cache
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@lru_cache(maxsize=1)
def load_model(model_name: str = "gpt2",
               device: str | torch.device = "cpu"):
    """Load model & tokenizer, cache for subsequent calls."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name,
                                                     output_hidden_states=True)
    model.to(device)
    model.eval()
    return tokenizer, model

def _last_hidden_states(text: str,
                        tokenizer,
                        model,
                        device: str | torch.device = "cpu") -> np.ndarray:
    with torch.no_grad():
        toks = tokenizer(text, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out  = model(**toks)
        # last_hidden_state shape: (1, seq_len, hidden)
        return out.hidden_states[-1][0].cpu().numpy()

def segment_vector(text: str,
                   tokenizer,
                   model,
                   device: str | torch.device = "cpu") -> np.ndarray:
    """Return 768‑d mean vector over tokens for *last* layer."""
    h = _last_hidden_states(text, tokenizer, model, device)
    return h.mean(axis=0)

def layerwise_segment_vectors(text: str,
                              tokenizer,
                              model,
                              device: str | torch.device = "cpu"
                             ) -> List[np.ndarray]:
    """Return list[n_layers+1] of (hidden,) means for every layer (incl. embed)."""
    with torch.no_grad():
        toks = tokenizer(text, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out  = model(**toks)
        return [layer[0].mean(dim=0).cpu().numpy()
                for layer in out.hidden_states]
