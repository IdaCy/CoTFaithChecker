from __future__ import annotations
import json, logging, math, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

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

_TEMPLATES = {
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
             "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "qwen": "User: {instruction}\nAssistant:",
}
def _chat_template(model_name: str):
    m = model_name.lower()
    if "llama" in m: return _TEMPLATES["llama"]
    if "qwen"  in m: return _TEMPLATES["qwen"]
    return "{instruction}"







import json, os, time, logging
from typing import List, Dict, Optional

import torch
from a_confirm_posthoc.utils.prompt_constructor import construct_prompt
from a_confirm_posthoc.parallelization.model_handler import generate_completion
from accelerate.utils import gather_object

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
q








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
        prompt_base: str,
        hint_text: Optional[str]=None,
        option_labels: List[str]=["A","B","C","D"],
        max_new_tokens: int=512,
        intervention_prompt="... The final answer is [",
        device=None):
    device = device or model.device
    tmpl  = _chat_template(getattr(model.config, "name_or_path", ""))
    prefix, suffix = tmpl.split("{instruction}")
    tok_prefix = tok.encode(prefix,  add_special_tokens=False)
    tok_suffix = tok.encode(suffix,  add_special_tokens=False)
    tok_q      = tok.encode(prompt_base, add_special_tokens=False)
    tok_hint   = tok.encode(hint_text, add_special_tokens=False) if hint_text else []

    input_ids = torch.tensor([tok_prefix + tok_q + tok_hint + tok_suffix],
                             device=device)
    attn_mask = torch.ones_like(input_ids)
    idx_q      = list(range(len(tok_prefix), len(tok_prefix)+len(tok_q)))
    idx_hint   = list(range(idx_q[-1]+1, idx_q[-1]+1+len(tok_hint)))
    idx_prompt = idx_q + idx_hint
    opt_ids    = _opt_token_ids(option_labels, tok)
    intv_ids   = tok.encode(intervention_prompt, add_special_tokens=False)

    generated, past_kv, out_rows, n_completed = [], None, [], 0
    with torch.no_grad():
        for tkn_step in range(max_new_tokens):
            out = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_attentions=True)
            logits = out.logits[:, -1, :]
            past_kv = out.past_key_values
            last_att = out.attentions[-1][0]

            next_id = int(logits.argmax(-1))
            generated.append(next_id)
            logging.debug("token %d generated id=%d", tkn_step+1, next_id)
            input_ids = torch.tensor([[next_id]], device=device)
            attn_mask = torch.ones_like(input_ids)

            decoded = tok.decode(generated, skip_special_tokens=True)
            sents = _split_sents(decoded)

            ends_with_punct = decoded.rstrip().endswith((".", "!", "?"))
            completed_cnt   = len(sents) if ends_with_punct else len(sents) - 1

            while n_completed < completed_cnt:
                sent_text = sents[n_completed]
                n_completed += 1

                att_prompt      = _avg_last_token_att(last_att, idx_prompt)
                att_prompt_only = (_avg_last_token_att(last_att, idx_q)
                                if hint_text else None)
                att_hint_only   = (_avg_last_token_att(last_att, idx_hint)
                                if hint_text else None)

                # fast logits: reuse KV-cache for intervention prompt #
                g_logits = model(
                    input_ids=torch.tensor([intv_ids], device=device),
                    past_key_values=past_kv,
                    use_cache=False).logits[0, -1] 
                ans_logits = {lbl: float(g_logits[tid].item())
                            for lbl, tid in opt_ids.items()}
                ans_probs  = _softmax(ans_logits)

                """out_rows.append(dict(
                    sentence_id=n_completed,
                    sentence=sent_text,
                    avg_att_prompt=att_prompt,
                    avg_att_prompt_only=att_prompt_only,
                    avg_att_hint_only=att_hint_only,
                    answer_logits=ans_logits,
                    answer_probs=ans_probs,
                ))"""
                
                out_rows.append(dict(
                    sentence_id=n_completed,
                    sentence=sent_text,
                    avg_att_prompt=att_prompt,
                    avg_att_prompt_only=att_prompt_only,
                    avg_att_hint_only=att_hint_only,
                    answer_logits=ans_logits,
                    answer_probs=ans_probs,
                ))

                if (n_completed % 5 == 0 and hasattr(model, "_tqdm_outer")):
                    model._tqdm_outer.set_postfix(
                        qid=getattr(model, "_current_qid", "?"),
                        sent=n_completed,
                        refresh=False)

            if "</think>" in decoded or n_completed >= 128:
                break

    return {"sentences": out_rows,
            "full_completion": tok.decode(generated, skip_special_tokens=True)}

def run_batch_from_files(
        model, tok,
        questions_file: str,
        hints_file: Optional[str]=None,
        whitelist_file: Optional[str]=None,
        output_file: str="sentence_level_results.json",
        max_questions: Optional[int]=None,
        **generation_kwargs):
    questions = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    
    whitelist_len = 0
    if whitelist_file and Path(whitelist_file).exists():
        whitelist = set(json.loads(Path(whitelist_file).read_text()))
        questions = [q for q in questions if q["question_id"] in whitelist]
        logging.info("filtered %d questions from %d",
                     len(questions), len(questions))
        whitelist_len = len(whitelist)
        
    #whitelist_len = len(whitelist) if (whitelist_file and Path(whitelist_file).exists()) else 0
    if max_questions:
        if (whitelist_len > 0):
            shortest_max = min(x for x in (whitelist_len, max_questions) if x is not None)
            questions = questions[:shortest_max]
        else:
            questions = questions[:max_questions]

    #if max_questions: questions = questions[:max_questions]

    hints_map = {}
    if hints_file and Path(hints_file).exists():
        hints_map = {h["question_id"]: h for h in
                     json.loads(Path(hints_file).read_text(encoding="utf-8"))}

    def make_prompt(q):
        opts = "\n".join(f"[ {k} ] {q[k]}" for k in ("A","B","C","D"))
        return (f"{q['question']}\n\n{opts}\n\n"
                "Please answer with the letter of the correct option (eg "
                "[ A ], [ B ], [ C ], [ D ])")

    outer_bar = tqdm(questions, desc="questions", unit="q", total=len(questions))

    results = []
    for q in outer_bar:
        # expose bar + qid to the inner function
        model._tqdm_outer  = outer_bar
        model._current_qid = q["question_id"]

        hint_obj  = hints_map.get(q["question_id"])
        hint_text = hint_obj["hint_text"] if hint_obj else None

        rec = run_sentence_level_inference(
                  model, tok,
                  prompt_base = make_prompt(q),
                  hint_text   = hint_text,
                  **generation_kwargs)

        results.append(dict(
            question_id       = q["question_id"],
            hint_option       = hint_obj.get("hint_option")       if hint_obj else None,
            is_correct_option = hint_obj.get("is_correct_option") if hint_obj else None,
            **rec
        ))

    for _attr in ("_tqdm_outer", "_current_qid"):
        if hasattr(model, _attr):
            delattr(model, _attr)

    Path(output_file).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logging.info("saved %d records to %s", len(results), output_file)
    return results

__all__ = ["load_model_and_tokenizer",
           "run_sentence_level_inference",
           "run_batch_from_files"]
