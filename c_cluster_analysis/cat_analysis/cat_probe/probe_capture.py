#!/usr/bin/env python
"""
Probe-capture runner
====================

• Splits a *pre-generated* chain-of-thought (CoT) into the exact same sentences
  you used for annotation.  
• Runs the model once per cumulative sentence, recording:

    – mean-pooled hidden states for every decoder layer (you can restrict this)  
    – (optional) attentions + logits, for continuity with earlier experiments

• Saves everything to JSON (small / human-readable) or a binary `.pt` file
  (efficient for large-scale runs).

Adapted from the original attention-logit script.
"""

from __future__ import annotations
import json, logging, math, os, re, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path: str):
    dev = _device()
    logging.info("loading %s on %s", model_path, dev)
    model_name = model_path.split("/")[-1]

    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()

    return model, tok, model_name, dev


KNOWN_CHAT_TEMPLATES = {
    "llama":  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "qwen":   "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
}

def get_chat_template(model_name: str) -> str:
    name = model_name.lower()
    if "llama" in name:      return KNOWN_CHAT_TEMPLATES["llama3"]
    if "qwen"  in name:      return KNOWN_CHAT_TEMPLATES["qwen"]
    logging.warning("no dedicated chat template for %s → using llama-style", model_name)
    return KNOWN_CHAT_TEMPLATES["llama"]

def _opt_token_ids(opts: List[str], tok):
    return {o: tok.encode(f" {o}", add_special_tokens=False)[0] for o in opts}

_SENT_RGX = re.compile(r"(?<=\.)\s+")

def _split_sents(txt: str) -> List[str]:
    """
    **Must** be bit-identical to the splitter used when the annotation JSONs
    were produced, so that sentence_id alignment holds.
    """
    start = txt.find("<think>");  end = txt.find("</think>")
    if start != -1: txt = txt[start + 7:]
    if end   != -1: txt = txt[:end]

    txt = re.sub(r"\s+", " ", txt.strip())
    parts = [p.strip() for p in _SENT_RGX.split(txt) if p.strip()]

    merged, i = [], 0
    while i < len(parts):
        # merge "1." + following clause
        if re.fullmatch(r"\d+\.", parts[i]) and i + 1 < len(parts):
            merged.append(f"{parts[i]} {parts[i+1]}")
            i += 2
        else:
            merged.append(parts[i]);  i += 1
    return merged


def _avg_last_token_att(att: torch.Tensor, idxs: List[int]) -> float:
    if not idxs:
        return 0.0
    return float(att[:, -1, idxs].mean().item())

def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    t = torch.tensor(list(logits.values()))
    p = torch.softmax(t, 0).tolist()
    return dict(zip(logits.keys(), p))

# ──────────────────────────────────────────────────────────────────────────────
#  Core: run_sentence_level_inference  (now with hidden-state capture)
# ──────────────────────────────────────────────────────────────────────────────
def run_sentence_level_inference(
    model,
    tok,
    *,
    prompt_base: str,                     # question + options (NO hint)
    hint_text: Optional[str],
    full_cot: str,
    option_labels: List[str] = ["A", "B", "C", "D"],
    intervention_prompt: str = "... The final answer is [",
    layers_to_keep: Optional[List[int]] = None,   # None → last layer only
    device=None,
    keep_attn_and_logits: bool = False,           # turn off to save space
):
    """
    Returns a dict with keys:
        • 'sentences' → list of per-sentence records
        • 'completion' → the original CoT text
    Each sentence-record contains:
        - sentence_id, sentence
        - pooled_hs: { layer_name: list[float] }
        - (optional) attentions + logits
    """
    device = device or model.device
    chat_prefix, chat_suffix = get_chat_template(
        getattr(model.config, "name_or_path", "")).split("{instruction}")

    # frozen tokens
    ids_prefix = tok.encode(chat_prefix, add_special_tokens=False)
    ids_q_opts = tok.encode(prompt_base,  add_special_tokens=False)
    ids_hint   = tok.encode(hint_text,    add_special_tokens=False) if hint_text else []
    ids_suffix = tok.encode(chat_suffix,  add_special_tokens=False)

    idx_q      = list(range(len(ids_prefix), len(ids_prefix) + len(ids_q_opts)))
    idx_hint   = list(range(idx_q[-1] + 1, idx_q[-1] + 1 + len(ids_hint)))
    idx_prompt = idx_q + idx_hint

    intv_ids   = tok.encode(intervention_prompt, add_special_tokens=False)
    opt_ids    = _opt_token_ids(option_labels, tok)

    sentences  = _split_sents(full_cot)
    out_rows   = []
    running_ids: List[int] = []          # reasoning tokens so far

    # we will need the individual sentence token sets to compute
    # start/end positions later
    for sid, sent in enumerate(sentences, 1):
        curr_sent_ids = tok.encode(sent, add_special_tokens=False)
        running_ids.extend(curr_sent_ids)

        seq_ids = torch.tensor(
            [ids_prefix + ids_q_opts + ids_hint + ids_suffix + running_ids],
            device=device,
        )

        with torch.no_grad():
            out = model(
                seq_ids,
                output_attentions = keep_attn_and_logits,
                output_hidden_states = True,   # ALWAYS true for probe capture
                use_cache = True,
            )

        hidden_states = out.hidden_states       # tuple: (emb, l1, l2, …, ln)

        # token-span in *this* forward pass that corresponds to the current sentence
        sent_start = len(ids_prefix) + len(ids_q_opts) + len(ids_hint) + len(ids_suffix) \
                   + (len(running_ids) - len(curr_sent_ids))
        sent_end   = sent_start + len(curr_sent_ids)

        # ── Pool hidden states ────────────────────────────────────────────────
        if layers_to_keep is None:
            layers_to_keep = [len(hidden_states) - 1]      # last layer index

        pooled_hs: Dict[str, torch.Tensor] = {}
        for li, h in enumerate(hidden_states[1:], 1):      # skip embeddings
            if li not in layers_to_keep:        # sparse capture
                continue
            pooled_hs[f"layer_{li}"] = (
                h[0, sent_start:sent_end].mean(0)
                  .to(torch.float16)            # ~2× smaller than fp32
                  .cpu()
            )

        # ── Optionally keep attention + logits (for continuity) ──────────────
        extra: Dict[str, Any] = {}

        if keep_attn_and_logits:
            last_att = out.attentions[-1][0]                 # heads × tgt × src
            extra["avg_att_prompt"]      = _avg_last_token_att(last_att, idx_prompt)
            extra["avg_att_prompt_only"] = _avg_last_token_att(last_att, idx_q) if hint_text else None
            extra["avg_att_hint_only"]   = _avg_last_token_att(last_att, idx_hint) if hint_text else None

            with torch.no_grad():
                logits = model(
                    input_ids = torch.tensor([intv_ids], device=device),
                    past_key_values = out.past_key_values,
                    use_cache = False,
                ).logits[0, -1]

            ans_logits = {lbl: float(logits[tid]) for lbl, tid in opt_ids.items()}
            extra["answer_logits"] = ans_logits
            extra["answer_probs"]  = _softmax(ans_logits)

        # ── Assemble record ───────────────────────────────────────────────────
        row = dict(
            sentence_id = sid,
            sentence    = sent,
            pooled_hs   = {k: v.numpy().tolist() for k, v in pooled_hs.items()},
            **extra
        )
        out_rows.append(row)

    return {
        "sentences":   out_rows,
        "completion":  full_cot,
    }

# ──────────────────────────────────────────────────────────────────────────────
#  Driver for many questions
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_from_files(
    model,
    tok,
    *,
    questions_file: str,
    full_cot_file: str,
    output_file: str = "sentence_level_hidden.json",
    layers_to_keep: Optional[List[int]] = None,
    keep_attn_and_logits: bool = False,
    max_questions: Optional[int] = None,
    whitelist_file: Optional[str] = None,
    construct_prompt_fn=None,          # injected dependency
):
    """
    Runs sentence-level inference for every question in `questions_file`,
    assuming you have already generated the *full* CoT and stored it in
    `full_cot_file`.
    """
    if construct_prompt_fn is None:
        # minimal replacement – no hint in the prompt
        def construct_prompt_fn(entry):
            return f"{entry['question']}\n\nOptions:\n" + \
                   "\n".join([f"[ {k} ] {entry[k]}" for k in "ABCD"]) + \
                   "\n\nPlease answer with the letter only."

    questions = json.loads(Path(questions_file).read_text("utf-8"))

    # whitelist handling
    if whitelist_file and Path(whitelist_file).exists():
        wl = set(int(x) for x in json.loads(Path(whitelist_file).read_text()))
        questions = [q for q in questions if int(q["question_id"]) in wl]
        logging.info("kept %d questions after whitelist", len(questions))

    if max_questions:
        questions = questions[:max_questions]

    cot_map = {c["question_id"]: c["completion"]
               for c in json.loads(Path(full_cot_file).read_text("utf-8"))}

    outer_bar = tqdm(questions, desc="questions", unit="q", total=len(questions))
    results: List[Dict[str, Any]] = []

    for q in outer_bar:
        model._tqdm_outer  = outer_bar
        model._current_qid = q["question_id"]

        prompt_base_no_hint = construct_prompt_fn({**q, "hint_text": None})
        full_cot = cot_map[q["question_id"]]          # must exist

        rec = run_sentence_level_inference(
            model,
            tok,
            prompt_base          = prompt_base_no_hint,
            hint_text            = None,
            full_cot             = full_cot,
            layers_to_keep       = layers_to_keep,
            keep_attn_and_logits = keep_attn_and_logits,
        )

        results.append(dict(
            question_id = q["question_id"],
            **rec
        ))

    for attr in ("_tqdm_outer", "_current_qid"):
        if hasattr(model, attr):
            delattr(model, attr)

    Path(output_file).write_text(json.dumps(results, indent=2), "utf-8")
    logging.info("saved %d records to %s", len(results), output_file)
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def run_probe_capture(
    model_path: str,
    questions_file: str,
    full_cot_file: str,
    *,
    output_file: str          = "sentence_level_hidden.json",
    layers: Optional[List[int]] = None,     # None → last only
    keep_attn_and_logits: bool = False,
    max_questions: Optional[int] = None,
    whitelist_file: Optional[str] = None,
):
    model, tok, _, _ = load_model_and_tokenizer(model_path)
    res = run_batch_from_files(
        model, tok,
        questions_file      = questions_file,
        full_cot_file       = full_cot_file,
        output_file         = output_file,
        layers_to_keep      = layers,
        keep_attn_and_logits= keep_attn_and_logits,
        max_questions       = max_questions,
        whitelist_file      = whitelist_file,
    )
    return res


if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--questions-file", required=True)
    ap.add_argument("--full-cot-file", required=True)
    ap.add_argument("--output-file", default="sentence_level_hidden.json")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="indices (1-based) of decoder layers to store "
                         "(omit → last layer only)")
    ap.add_argument("--keep-attn-and-logits", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--whitelist-file")
    args = ap.parse_args()

    run_probe_capture(
        model_path           = args.model_path,
        questions_file       = args.questions_file,
        full_cot_file        = args.full_cot_file,
        output_file          = args.output_file,
        layers               = args.layers,
        keep_attn_and_logits = args.keep_attn_and_logits,
        max_questions        = args.max_questions,
        whitelist_file       = args.whitelist_file,
    )
