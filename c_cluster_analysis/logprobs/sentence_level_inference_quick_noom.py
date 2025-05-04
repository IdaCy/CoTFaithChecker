from __future__ import annotations
import json, logging, math, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

import json, os, time, logging
from typing import List, Dict, Optional

import torch
from a_confirm_posthoc.main.prompt_constructor import construct_prompt
from a_confirm_posthoc.parallelization.model_handler import generate_completion
from accelerate.utils import gather_object

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def _device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path: str):
    dev = _device()
    logging.info("loading %s on %s", model_path, dev)
    model_name = model_path.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
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
    if "llama" in name:
        return KNOWN_CHAT_TEMPLATES["llama3"]
    if "qwen" in name:
        return KNOWN_CHAT_TEMPLATES["qwen"]
    logging.warning(f"No specific chat template for {model_name}; using llama.")
    return KNOWN_CHAT_TEMPLATES["llama"]

def _opt_token_ids(opts: List[str], tok):
    return {o: tok.encode(f" {o}", add_special_tokens=False)[0] for o in opts}

_SENT_RGX = re.compile(r"(?<=\.)\s+")

def _split_sents(txt: str) -> List[str]:
    start = txt.find("<think>");  end = txt.find("</think>")
    if start != -1: txt = txt[start + 7:]
    if end   != -1: txt = txt[:end]
    txt = re.sub(r"\s+", " ", txt.strip())
    parts = [p.strip() for p in _SENT_RGX.split(txt) if p.strip()]
    merged, i = [], 0
    while i < len(parts):
        if re.fullmatch(r"\d+\.", parts[i]) and i+1 < len(parts):
            merged.append(f"{parts[i]} {parts[i+1]}");  i += 2
        else:
            merged.append(parts[i]);  i += 1
    return merged

def _avg_last_token_att(att: torch.Tensor, idxs: List[int]) -> float:
    if not idxs: return 0.0
    return float(att[:, -1, idxs].mean().item())

def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    import torch
    t = torch.tensor(list(logits.values()))
    p = torch.softmax(t, 0).tolist()
    return dict(zip(logits.keys(), p))

def run_sentence_level_inference(
        model, tok,
        prompt_base: str,            # question + options   (NO hint)
        hint_text: Optional[str],    # hint, or None
        full_cot: str,               # the already-generated chain-of-thought
        option_labels: List[str] = ["A", "B", "C", "D"],
        intervention_prompt: str = "... The final answer is [",
        device=None):

    device = device or model.device
    chat_prefix, chat_suffix = get_chat_template(
        getattr(model.config, "name_or_path", "")).split("{instruction}")

    # tokens that never change 
    ids_prefix = tok.encode(chat_prefix, add_special_tokens=False)
    ids_q_opts = tok.encode(prompt_base,  add_special_tokens=False)
    ids_hint   = tok.encode(hint_text,    add_special_tokens=False) if hint_text else []
    ids_suffix = tok.encode(chat_suffix,  add_special_tokens=False)
    idx_q      = list(range(len(ids_prefix), len(ids_prefix) + len(ids_q_opts)))
    idx_hint   = list(range(idx_q[-1] + 1, idx_q[-1] + 1 + len(ids_hint)))
    idx_prompt = idx_q + idx_hint
    intv_ids   = tok.encode(intervention_prompt, add_special_tokens=False)
    opt_ids    = _opt_token_ids(option_labels, tok)

    # sentence list
    sentences = _split_sents(full_cot)
    out_rows  = []
    running_ids: List[int] = []           # reasoning tokens accumulated

    for sid, sent in enumerate(sentences, 1):
        # 0) append the current sentence to the running context
        running_ids.extend(tok.encode(sent, add_special_tokens=False))
        seq_ids = torch.tensor(
            [ids_prefix + ids_q_opts + ids_hint + ids_suffix + running_ids],
            device=device)

        # 1) forward pass to get attentions on the last reasoning token
        with torch.no_grad():
            out = model(seq_ids, output_attentions=True, use_cache=False)

        last_att = out.attentions[-1][0].clone()       # heads × tgt × src
        del out                                        # free ~90 % of the memory
        torch.cuda.empty_cache()

        att_prompt      = _avg_last_token_att(last_att, idx_prompt)
        att_prompt_only = _avg_last_token_att(last_att, idx_q)     if hint_text else None
        att_hint_only   = _avg_last_token_att(last_att, idx_hint)  if hint_text else None

        # 2) second (tiny) pass: add the intervention prompt, get option logits
        seq_ids_plus = torch.tensor(
            [seq_ids[0].tolist() + intv_ids], device=device)

        with torch.no_grad():
            logits = model(seq_ids_plus,
                        output_attentions=False,
                        use_cache=False).logits[0, -1]

        ans_logits = {lbl: float(logits[tid]) for lbl, tid in opt_ids.items()}
        ans_probs  = _softmax(ans_logits)

        out_rows.append(dict(
            sentence_id         = sid,
            sentence            = sent,
            avg_att_prompt      = att_prompt,
            avg_att_prompt_only = att_prompt_only,
            avg_att_hint_only   = att_hint_only,
            answer_logits       = ans_logits,
            answer_probs        = ans_probs,
        ))

    return {
        "sentences":       out_rows,
        "completion": full_cot
    }


def run_batch_from_files(
        model, tok,
        questions_file: str,
        hints_file: Optional[str]        = None,
        full_cot_file: Optional[str]     = None,    # ← NEW
        whitelist_file: Optional[str]    = None,
        output_file: str = "sentence_level_results.json",
        max_questions: Optional[int]     = None):

    questions = json.loads(Path(questions_file).read_text("utf-8"))

    if whitelist_file and Path(whitelist_file).exists():
        wl = set(int(x) for x in json.loads(Path(whitelist_file).read_text()))
        questions = [q for q in questions if int(q["question_id"]) in wl]
        logging.info("kept %d questions after whitelist", len(questions))

    if max_questions:
        questions = questions[:max_questions]

    hints_map = {}
    if hints_file and Path(hints_file).exists():
        hints_map = {h["question_id"]: h
                     for h in json.loads(Path(hints_file).read_text("utf-8"))}

    cot_map = {}
    if full_cot_file and Path(full_cot_file).exists():
        cot_map = {c["question_id"]: c["completion"]
                   for c in json.loads(Path(full_cot_file).read_text("utf-8"))}

    outer_bar = tqdm(questions, desc="questions", unit="q", total=len(questions))
    results   = []

    for q in outer_bar:
        model._tqdm_outer  = outer_bar
        model._current_qid = q["question_id"]

        hint_obj  = hints_map.get(q["question_id"])
        hint_text = hint_obj.get("hint_text") if hint_obj else None

        # build prompt (NO hint) for attentions
        entry_no_hint       = {**q, "hint_text": None}
        prompt_base_no_hint = construct_prompt(entry_no_hint)

        full_cot = cot_map[q["question_id"]]          # already generated

        rec = run_sentence_level_inference(
                model, tok,
                prompt_base = prompt_base_no_hint,
                hint_text   = hint_text,
                full_cot    = full_cot)

        results.append(dict(
            question_id       = q["question_id"],
            hint_option       = hint_obj.get("hint_option")       if hint_obj else None,
            is_correct_option = hint_obj.get("is_correct_option") if hint_obj else None,
            **rec
        ))

    for attr in ("_tqdm_outer", "_current_qid"):
        if hasattr(model, attr):
            delattr(model, attr)

    Path(output_file).write_text(json.dumps(results, indent=2), "utf-8")
    logging.info("saved %d records to %s", len(results), output_file)
    return results


__all__ = ["load_model_and_tokenizer",
           "run_sentence_level_inference",
           "run_batch_from_files"]
