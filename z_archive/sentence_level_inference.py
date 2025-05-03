# ──────────────────────────────────────────────────────────────────────────────
#  File: src/sentence_level_inference.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Sentence-level chain-of-thought inference

• Generates text one *complete* sentence at a time (sentences are detected with
  the exact `_split_sentences()` rule you already use for Gem­ini annotation).
• After every sentence it
    1. records the sentence text,
    2. records the model’s **guess so far** (logits over MCQ option tokens),
    3. stores the **average last-token attention** paid to
          – the whole prompt,
          – the prompt part only (question + options + instruction),
          – the hint part only (if present).

Everything is batched inside a single function
`run_sentence_level_inference( … )` so you can plug it straight into a loop
over many questions.

The file also re-exports a few utilities you already use so that a notebook
only needs to import *this* file.
"""
from __future__ import annotations
import math, re, logging, os, sys, json
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ╭──────────────────────── Existing helpers (unedited) ─────────────────────╮ #
# ▸ model / tokenizer loader
def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str, torch.device]:
    device = _get_device()
    logging.info("Loading model %s on %s", model_path, device)
    model_name = model_path.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok, model_name, device

# ▸ chat-template selector (simplified – same logic as before)
_CHAT_TEMPLATES = {
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
             "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "qwen":  "User: {instruction}\nAssistant:",
}
def get_chat_template(model_name: str) -> str:
    ml = model_name.lower()
    if "llama" in ml: return _CHAT_TEMPLATES["llama"]
    if "qwen"  in ml: return _CHAT_TEMPLATES["qwen"]
    return "{instruction}"

# ▸ MCQ option-token mapping (unchanged)
def get_option_token_ids(options: List[str], tok: AutoTokenizer
                         ) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lbl in options:
        tid = tok.encode(f" {lbl}", add_special_tokens=False)[0]
        out[lbl] = tid
    return out
# ╰────────────────────────────────────────────────────────────────────────────╯ #

# ▸ exact same splitter you use for the annotator
_SENT_SPLIT_RGX = re.compile(r"(?<=\.)\s+")
def _split_sentences(txt: str) -> List[str]:
    start = txt.find("<think>")
    if start != -1:
        txt = txt[start + len("<think>") :]
    end = txt.find("</think>")
    if end != -1:
        txt = txt[:end]
    txt = re.sub(r"\s+", " ", txt.strip())
    parts = [p.strip() for p in _SENT_SPLIT_RGX.split(txt) if p.strip()]

    merged: List[str] = []
    i = 0
    while i < len(parts):
        if re.fullmatch(r"\d+\.", parts[i]) and i + 1 < len(parts):
            merged.append(f"{parts[i]} {parts[i+1]}")
            i += 2
        else:
            merged.append(parts[i]); i += 1
    return merged

# ╭──────────────────────────── New helper utils ─────────────────────────────╮ #
def _average_last_token_attention(
    att: torch.Tensor,                       # shape: [heads, seq, seq]
    seg_idxs: List[int]
) -> float:
    if not seg_idxs: return 0.0
    with torch.no_grad():
        last_tok_vec = att[:, -1, :]         # [heads, seq]
        val = last_tok_vec[:, seg_idxs].mean().item()
    return float(val)

def _detect_sentence_increment(
    full_generated: str, prev_n_sentences: int
) -> Tuple[Optional[str], int]:
    """Return newly completed sentence (if any) and updated count."""
    sents = _split_sentences(full_generated)
    if len(sents) > prev_n_sentences:
        return sents[-1], len(sents)
    return None, prev_n_sentences
# ╰────────────────────────────────────────────────────────────────────────────╯ #

# ╭─────────────────────── Core incremental-generation fn ────────────────────╮ #
def run_sentence_level_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_base: str,
    hint_text: Optional[str] = None,
    option_labels: List[str] = ["A","B","C","D"],
    max_new_tokens: int = 512,
    intervention_prompt: str = "... The final answer is [",
    *,
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    """
    Returns
    -------
    {
      "sentences": [
        {
          "text": str,
          "avg_att_prompt": float,
          "avg_att_prompt_only": float | None,
          "avg_att_hint_only": float | None,
          "answer_logits": { "A": float, ... }
        }, ...
      ],
      "full_completion": str
    }
    """
    device = device or model.device
    # ── prepare chat prompt and token-index bookkeeping ────────────────────
    instr = prompt_base + (f"\n\n{hint_text}" if hint_text else "")
    tmpl  = get_chat_template(getattr(model.config, "name_or_path", ""))
    parts = tmpl.split("{instruction}")
    if len(parts) != 2:
        raise ValueError("Chat template must contain '{instruction}' once")
    prefix, suffix = parts
    tok_prefix = tokenizer.encode(prefix,  add_special_tokens=False)
    tok_suffix = tokenizer.encode(suffix,  add_special_tokens=False)
    tok_prompt_only = tokenizer.encode(prompt_base, add_special_tokens=False)
    tok_hint = tokenizer.encode(hint_text, add_special_tokens=False) if hint_text else []

    input_ids = torch.tensor(
        [ tok_prefix + tok_prompt_only + tok_hint + tok_suffix ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, device=device)

    # segmentation indices (in *current* sequence – grows with generation)
    idx_prompt_only = list(range(len(tok_prefix),
                                 len(tok_prefix)+len(tok_prompt_only)))
    idx_hint        = list(range(idx_prompt_only[-1]+1,
                                 idx_prompt_only[-1]+1+len(tok_hint)))
    idx_whole_prompt = idx_prompt_only + idx_hint

    opt_token_ids = get_option_token_ids(option_labels, tokenizer)

    past_kv = None
    generated: List[int] = []
    results: List[Dict[str, Any]] = []
    n_sents = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_attentions=True)
            logits = out.logits[:, -1, :]                     # [1, vocab]
            past_kv = out.past_key_values
            last_layer_att = out.attentions[-1][0]            # [heads, seq, seq]

            # greedy (do_sample=False in outer loop earlier) – use argmax
            next_id = int(torch.argmax(logits, dim=-1))
            generated.append(next_id)

            # extend input for next step
            input_ids = torch.tensor([[next_id]], device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

            # decode so far and check if we completed a new sentence
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            new_sent, n_sents = _detect_sentence_increment(decoded, n_sents)
            if new_sent is None:
                continue  # need more tokens to finish a sentence

            # ── attention metrics (last-token to segments) ────────────────
            att_prompt   = _average_last_token_attention(last_layer_att,
                                                         idx_whole_prompt)
            att_prompt_only = (_average_last_token_attention(last_layer_att,
                                                             idx_prompt_only)
                               if hint_text else None)
            att_hint_only  = (_average_last_token_attention(last_layer_att,
                                                            idx_hint)
                              if hint_text else None)

            # ── “guess so far” logits for MCQ options ────────────────────
            # build {prompt context + generated SoT + intervention_prompt}
            seq_for_guess = (tok_prefix + tok_prompt_only + tok_hint +
                             tok_suffix + generated +
                             tokenizer.encode(intervention_prompt,
                                              add_special_tokens=False))
            guess_ids = torch.tensor([seq_for_guess], device=device)
            guess_logits = model(input_ids=guess_ids).logits[:, -1, :]
            ans_logits = {lbl: float(guess_logits[0, tid].item())
                          for lbl, tid in opt_token_ids.items()}

            results.append({
                "text": new_sent,
                "avg_att_prompt": att_prompt,
                "avg_att_prompt_only": att_prompt_only,
                "avg_att_hint_only": att_hint_only,
                "answer_logits": ans_logits
            })

            # optional early stop if model already produced </think>
            if "</think>" in decoded:
                break

    full_comp = tokenizer.decode(generated, skip_special_tokens=True)
    return {"sentences": results, "full_completion": full_comp}
# ╰─────────────────────────────────────────────────────────────────────────────╯ #

__all__ = [
    "load_model_and_tokenizer",
    "get_chat_template",
    "get_option_token_ids",
    "run_sentence_level_inference",
]
