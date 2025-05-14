# cot_steer_utils5.py  ←  new version
# ---------------------------------------------------------------
"""
Vector-steering utilities (safe for LoRA / flash-attn / RoPE models).

Key changes
===========

* _SteeringHook now **edits the tensor in-place** instead of returning
  a brand-new one, so strides stay untouched.

* dtype is matched with `hidden.dtype`; no more float16⇄bfloat16 surprises.

* a small helper `register_steering_hooks(...)` keeps the notebook tidy.
"""

from __future__ import annotations
import json, logging, re, math
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_attr_vectors(attr_dir: Path | str) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return {category: {layer_name: tensor(hidden_size)}}"""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for fp in Path(attr_dir).glob("*.pt"):
        cat = fp.stem.replace("_", " ")
        out[cat] = torch.load(fp, map_location="cpu")
    return out

# ─── text helpers ───────────────────────────────────────────────────
_SENT_RGX = re.compile(r"(?<=\.)\s+")

def _split_sentences(txt: str) -> List[str]:
    s = txt.find("<think>"); e = txt.find("</think>")
    if s != -1: txt = txt[s + 7:]
    if e != -1: txt = txt[:e]
    txt = re.sub(r"\s+", " ", txt.strip())
    parts = [p.strip() for p in _SENT_RGX.split(txt) if p.strip()]
    merged, i = [], 0
    while i < len(parts):
        if re.fullmatch(r"\d+\\.", parts[i]) and i + 1 < len(parts):
            merged.append(f"{parts[i]} {parts[i+1]}"); i += 2
        else:
            merged.append(parts[i]); i += 1
    return merged

# ─── steering hook ─────────────────────────────────────────────────
# --- in cot_steer_utils5.py (or 6) ---------------------------------
class _SteeringHook:
    """
    Adds α·vector to the layer output, returns a **fully new, contiguous**
    tensor so downstream `view` / `reshape` calls never trip on strides.
    """
    def __init__(self, vec: torch.Tensor, alpha: float):
        self.vec   = vec          # (H,)
        self.alpha = alpha

    def __call__(self, _mod, _inp, out):
        if isinstance(out, tuple):
            hidden, *rest = out
        else:
            hidden, rest = out, []

        # allocate a fresh contiguous tensor (safe, cheap – a single memcpy)
        steer = self.vec.to(hidden.device, dtype=hidden.dtype)
        for _ in range(hidden.dim() - 1):        # expand to (1, …, H)
            steer = steer.unsqueeze(0)

        new_hidden = (hidden + self.alpha * steer).contiguous()

        # if the layer returns a tuple, patch element-0 in-place
        if rest:
            out = (new_hidden, *rest)
        else:
            out = new_hidden
        return out


# convenience wrapper – keeps the notebook clean
def register_steering_hooks(model, steer_vectors: Dict[str, torch.Tensor],
                            alpha: float) -> Dict[str, torch.utils.hooks.RemovableHandle]:
    """
    `steer_vectors` = {"layer_11": tensor(4096), …}
    Returns dict of RemovableHandle so the caller can `.remove()` afterwards.
    """
    handles = {}
    for ln, vec in steer_vectors.items():
        li = int(ln.split("_")[1]) - 1          # to 0-based index
        layer = model.model.layers[li]
        vec   = vec.to(model.device, dtype=next(layer.parameters()).dtype)
        handles[ln] = layer.register_forward_hook(_SteeringHook(vec, alpha))
    return handles

# ─── generation & attention helpers ────────────────────────────────
def _softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def _avg_last_token_att(att: torch.Tensor, idxs: List[int]) -> float:
    if not idxs: return 0.0
    return float(att[:, -1, idxs].mean().item())

def _opt_token_ids(opts, tok):
    return {o: tok.encode(f" {o}", add_special_tokens=False)[0] for o in opts}

# ─── main experiment driver ────────────────────────────────────────
def run_steering_experiment(
    *,
    model,
    tok,
    steer_vectors: Dict[str, torch.Tensor],
    cat_target: str,
    alphas: List[float],
    questions_file: Path | str,
    full_cot_file: Path | str,
    output_file: Path | str,
    hints_file: Optional[Path | str] = None,
    option_labels: List[str] = ["A", "B", "C", "D"],
    intervention_prompt: str = "... The final answer is [",
    max_questions: Optional[int] = None,
):
    """
    For each question:
      • regenerate the *first* sentence of `cat_target`
        with every `alpha` (0 ⇒ original);
      • capture attention + answer logits/probs.

    All heavy lifting (vector injection, token-level generation, attention
    maths) happens here; the notebook stays a 10-line shell.
    """
    # ---------- load data -------------------------------------------------
    qs  = json.loads(Path(questions_file).read_text("utf-8"))
    if max_questions: qs = qs[:max_questions]

    cot_map  = {c["question_id"]: c["completion"]
                for c in json.loads(Path(full_cot_file).read_text("utf-8"))}

    hints_map = {}
    if hints_file and Path(hints_file).exists():
        hints_map = {h["question_id"]: h for h in
                     json.loads(Path(hints_file).read_text("utf-8"))}

    chat_template = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                     "{instruction}<|eot_id|><|start_header_id|>assistant"
                     "<|end_header_id|>\n")

    # ---------- results container ----------------------------------------
    out = []

    outer = tqdm(qs, desc="steering", unit="q")
    for q in outer:
        qid = q["question_id"]
        cot = cot_map[qid]
        sents = _split_sentences(cot)

        # naive: assume sentence-id ordering == annotation ordering.
        # regenerate *first* sentence of the target category
        target_sent = sents[0]
        for s in sents:
            if cat_target.replace(" ", "_").lower() in s.lower():
                target_sent = s
                break

        # prompt ids
        question_block = (
            q["question"] + "\n\nOptions:\n" +
            "\n".join([f"[ {k} ] {q[k]}" for k in "ABCD"]) +
            "\n\nPlease answer with the letter only."
        )
        hint_text = hints_map.get(qid, {}).get("hint_text")
        pre, post = chat_template.split("{instruction}")
        ids = (tok.encode(pre,  add_special_tokens=False) +
               tok.encode(question_block, add_special_tokens=False) +
               (tok.encode(hint_text, add_special_tokens=False) if hint_text else []) +
               tok.encode(post, add_special_tokens=False))

        # run once per α ---------------------------------------------------
        sent_records = []
        for a in alphas:
            hooks = {}
            if a != 0.0:                       # zero ⇒ no hook at all
                hooks = register_steering_hooks(model, steer_vectors, a)
            # greedy generate until the first '.' + space after <think>
            gen_ids = []
            for _ in range(128):
                input_ids = torch.tensor([ids + gen_ids], device=model.device)
                with torch.no_grad():
                    nxt = int(model(input_ids, use_cache=True).logits[0, -1].argmax())
                gen_ids.append(nxt)
                txt = tok.decode(gen_ids)
                if re.search(r"\.\s", txt):                 # end-of-sentence
                    break
            for h in hooks.values():
                h.remove()

            full_ids = ids + gen_ids
            with torch.no_grad():
                out_all = model(torch.tensor([full_ids], device=model.device),
                                output_attentions=True)
            last_att = out_all.attentions[-1][0]

            # attention summaries
            prompt_len = len(ids)
            idx_prompt = list(range(prompt_len))
            att_prompt = _avg_last_token_att(last_att, idx_prompt)

            # answer logits
            intv_ids = tok.encode(intervention_prompt, add_special_tokens=False)
            opt_ids  = _opt_token_ids(option_labels, tok)
            with torch.no_grad():
                logits = model(torch.tensor([full_ids + intv_ids],
                                            device=model.device)).logits[0, -1]
            logits_dict = {k: float(logits[v]) for k, v in opt_ids.items()}
            probs = _softmax_np(list(logits_dict.values()))
            probs_dict = dict(zip(option_labels, probs))

            sent_records.append(dict(
                alpha              = a,
                generated_sentence = tok.decode(gen_ids).strip(),
                avg_att_prompt     = att_prompt,
                answer_logits      = logits_dict,
                answer_probs       = probs_dict,
            ))

        out.append(dict(
            question_id     = qid,
            target_sentence = target_sent,
            results         = sent_records,
        ))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(out, indent=2), "utf-8")
    logging.info("saved %d records → %s", len(out), output_file)
    return out
