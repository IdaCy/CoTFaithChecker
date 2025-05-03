"""
Sentence-level extraction of
  • option logits
  • average prompt / question / hint attention
after every generated sentence of a chain-of-thought.
"""
from __future__ import annotations
import re, math, logging
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from .utils import get_intervention_prompt


_SENT_RE = re.compile(r"(?<=\.)\s+")
def split_sentences(text: str) -> List[str]:
    """Exact same rule that the LLM-annotator uses."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]

    # merge numbered list items (“1.” + sentence) – identical to annotator
    merged: List[str] = []
    i = 0
    while i < len(parts):
        if re.fullmatch(r"\d+\.", parts[i]) and i + 1 < len(parts):
            merged.append(f"{parts[i]} {parts[i + 1]}")
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return merged


def _mean_last_token_attention(
    attn_tensors: tuple[torch.Tensor, ...],   # (n_layers, 1, n_heads, seq, seq)
    target_idx: torch.Tensor                  # (n_targets,)
) -> float:
    """Average over layers, heads and the target token set."""
    # gather the attention rows that belong to the LAST query token
    per_layer = []
    for layer in attn_tensors:                       # shape: (1, n_heads, S, S)
        heads = layer[0, :, -1, :]                   # (n_heads, S)
        per_layer.append(heads.mean(0))             # (S,)
    stacked = torch.stack(per_layer)                # (n_layers, S)
    vec = stacked.mean(0)                           # (S,)
    return vec[target_idx].mean().item() if target_idx.numel() else 0.0


def _hint_token_indices(prompt: str,
                        hint_text: Optional[str],
                        tokenizer) -> tuple[list[int], list[int]]:
    """
    Returns (question_part_indices, hint_indices) inside the *prompt* token
    sequence.  If no hint is present both lists are empty.
    """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not hint_text:
        return list(range(len(prompt_tokens))), []   # whole prompt only

    # character offsets of the hint inside the prompt
    p = prompt.rfind(hint_text)
    if p == -1:
        # cannot find hint string – silently fall back
        return list(range(len(prompt_tokens))), []

    hint_prefix = prompt[:p]
    q_part_len  = len(tokenizer.encode(hint_prefix, add_special_tokens=False))
    hint_len    = len(tokenizer.encode(hint_text,   add_special_tokens=False))

    question_part = list(range(q_part_len))
    hint_part     = list(range(q_part_len, q_part_len + hint_len))
    return question_part, hint_part


def extract_sentence_sequence(
    completion_text: str,
    reasoning_start: int,
    reasoning_end: int,
    option_token_ids: dict[str, int],
    *,
    intervention_type: str,
    model,
    tokenizer,
    device: str,
    full_prompt_text: str,
    hint_text: Optional[str] = None
) -> list[dict]:
    """
    Returns one dict per generated sentence with:
       • "sentence"              – the text of that sentence
       • "logprobs"              – logits for every MCQ option
       • "avg_attn_prompt"       – mean attention to entire prompt
       • "avg_attn_question"     – mean attention to prompt *without* hint (None if no hint)
       • "avg_attn_hint"         – mean attention to hint (None if no hint)
    """

    # tokenise immutable pieces once
    prompt_tokens = tokenizer.encode(full_prompt_text,
                                     return_tensors="pt",
                                     add_special_tokens=False).to(device)
    question_indices, hint_indices = _hint_token_indices(
        full_prompt_text, hint_text, tokenizer
    )
    question_idx_t = torch.tensor(question_indices, device=device)
    hint_idx_t     = torch.tensor(hint_indices,     device=device)

    # intervention postfix ("... The final answer is [", …)
    intervention_prompt = get_intervention_prompt(intervention_type)
    intervention_tokens = tokenizer.encode(intervention_prompt,
                                           return_tensors="pt",
                                           add_special_tokens=False).to(device)

    # split CoT into the same sentences annotator will see
    reasoning_text = completion_text[reasoning_start: reasoning_end]
    sentences = split_sentences(reasoning_text)
    if not sentences:
        logging.warning("No sentences extracted - skip")
        return []

    # running list of token ids for the prefix that has already been “generated”
    prefix_token_ids: list[int] = []
    sequence: list[dict] = []

    for sent_idx, sent in enumerate(sentences, 1):
        # keep the *exact* whitespace the annotator expects
        sent_text = (" " if sent_idx > 1 else "") + sent
        sent_toks = tokenizer.encode(sent_text, add_special_tokens=False)
        prefix_token_ids.extend(sent_toks)

        # build input =  prompt  +  generated_prefix(+sent)  +  intervention
        prefix_tensor = torch.tensor([prefix_token_ids],
                                     device=device, dtype=torch.long)
        input_ids = torch.cat([prompt_tokens, prefix_tensor,
                               intervention_tokens], dim=1)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            output_attentions=True)
            logits      = outputs.logits[:, -1, :]          # (1, V)
            attentions  = outputs.attentions                # tuple(layer)

        step_logits: Dict[str, float] = {}
        for label, tok_id in option_token_ids.items():
            step_logits[label] = logits[0, tok_id].item()

        tgt_prompt = torch.arange(prompt_tokens.shape[1], device=device)
        avg_prompt   = _mean_last_token_attention(attentions, tgt_prompt)
        avg_question = (_mean_last_token_attention(attentions, question_idx_t)
                        if question_idx_t.numel() else None)
        avg_hint     = (_mean_last_token_attention(attentions, hint_idx_t)
                        if hint_idx_t.numel() else None)

        sequence.append({
            "sentence_id":      sent_idx,
            "sentence":         sent.strip(),
            "token_index":      len(prefix_token_ids),
            "logprobs":         step_logits,
            "avg_attn_prompt":  avg_prompt,
            "avg_attn_question":avg_question,
            "avg_attn_hint":    avg_hint
        })

    return sequence
