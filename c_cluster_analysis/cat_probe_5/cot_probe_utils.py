"""
cot_probe_utils.py
==================
Utility functions for probing chain-of-thought (CoT) categories.

•  load_model_and_tokenizer … HF-style loader (bfloat16, device-map="auto")
•  gather_category_sentences … find (question_id, sentence_id) pairs whose
   *top-confidence* category is in `main_categories`
•  run_probe_capture_for_categories … capture mean-pooled hidden states for
   those sentences (layers selectable; default = last)
•  train_linear_probes … train a logistic regression probe per layer
•  save_attribute_vectors … write ⟨category,layer⟩ mean vectors as *.pt files
---------------------------------------------------------------------------
This file is *stand-alone* – no project-specific imports required.
"""

from __future__ import annotations
import json, logging, random, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------- I/O
CATEGORY_NAMES = [
    "problem_restating","knowledge_augmentation","assumption_validation",
    "logical_deduction","option_elimination","uncertainty_or_certainty_expression",
    "backtracking","forward_planning","decision_confirmation",
    "answer_reporting","option_restating","other",
]

# ------------------------------ 1.  Model & sentence splitter ----------

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path: str):
    """Load HF model/tokenizer (bfloat16, device-map='auto')."""
    dev = _device()
    logging.info("Loading %s on %s", model_path, dev)
    tok   = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok, model.config.hidden_size, dev


_SENT_RGX = re.compile(r"(?<=\.)\s+")

def _split_think_sentences(txt: str) -> List[str]:
    """Extract the <think> … </think> region and split exactly like in annotation."""
    # isolate think-block
    s = txt.find("<think>")
    e = txt.find("</think>")
    if s != -1: txt = txt[s + 7:]
    if e != -1: txt = txt[:e]
    txt = re.sub(r"\s+", " ", txt.strip())

    parts = [p.strip() for p in _SENT_RGX.split(txt) if p.strip()]

    # merge numeric list heads (e.g. "1." + following text)
    merged, i = [], 0
    while i < len(parts):
        if re.fullmatch(r"\d+\.", parts[i]) and i + 1 < len(parts):
            merged.append(f"{parts[i]} {parts[i+1]}");  i += 2
        else:
            merged.append(parts[i]);  i += 1
    return merged


# ------------------------------ 2.  Annotation processing --------------

def gather_category_sentences(
    category_file: Path | str,
    *,
    main_categories: List[str],
    whitelist: Optional[Path | str] = None,
    max_samples: Optional[int]      = None,
) -> Dict[int, List[Tuple[int, str]]]:
    """
    Return {question_id: [(sentence_id, label), …]} keeping only sentences
    whose *highest-confidence* label lies in `main_categories`.
    """
    data = json.loads(Path(category_file).read_text("utf-8"))
    w_set = None
    if whitelist and Path(whitelist).exists():
        w_set = set(json.loads(Path(whitelist).read_text()))
        logging.info("Whitelist loaded – %d question_ids kept", len(w_set))

    sel: Dict[int, List[Tuple[int, str]]] = {}
    for entry in data:
        qid = entry["question_id"]
        if w_set and qid not in w_set:   # skip
            continue
        for ann in entry["annotations"]:
            # highest-confidence category for this sentence
            top_cat = max(CATEGORY_NAMES, key=lambda c: ann[c])
            if top_cat in main_categories:
                sel.setdefault(qid, []).append((ann["sentence_id"], top_cat))

    # deterministic “take-the-first-N” subset
    if max_samples and len(sel) > max_samples:
        sel = dict(list(sel.items())[:max_samples])
        logging.info("Kept the first %d questions (max_samples)", len(sel))


    return sel


# ------------------------------ 3.  Inference utilities -----------------

_CHAT_TEMPLATES = {
    "llama":  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "qwen":   "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
}
def _get_chat_template(model_name: str):
    name = model_name.lower()
    if "llama" in name: return _CHAT_TEMPLATES["llama"]
    if "qwen" in name:  return _CHAT_TEMPLATES["qwen"]
    return _CHAT_TEMPLATES["llama"]    # safe default

def _construct_prompt(question_block: str, template: str) -> str:
    return template.replace("{instruction}", question_block)


def _mean_pool(h: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Mean-pool hidden states [start:end) over tokens."""
    return h[0, start:end].mean(0)

# ------------------------------------------------------------------------
def _run_sentence_level_inference(
    model,
    tok,
    *,
    prompt_base: str,
    full_cot: str,
    layers_to_keep: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Return list of {'sentence_id', 'pooled_hs' {layer_name: vec}, 'sentence'}.
    Hidden-state vectors are *float16* numpy arrays to save RAM.
    """
    chat_prefix, chat_suffix = _get_chat_template(
        getattr(model.config, "name_or_path", "")
    ).split("{instruction}")

    # frozen tokens (question/prompt)
    ids_prefix = tok.encode(chat_prefix, add_special_tokens=False)
    ids_q      = tok.encode(prompt_base, add_special_tokens=False)
    ids_suffix = tok.encode(chat_suffix, add_special_tokens=False)

    sentences  = _split_think_sentences(full_cot)
    out_rows   = []
    running_ids: List[int] = []

    for sid, sent in enumerate(sentences, 1):
        sent_ids = tok.encode(sent, add_special_tokens=False)
        running_ids.extend(sent_ids)

        seq_ids = torch.tensor([ids_prefix + ids_q + ids_suffix + running_ids],
                               device=model.device)

        with torch.no_grad():
            out = model(seq_ids,
                        output_hidden_states=True,
                        use_cache=True)

        h_states = out.hidden_states           # (emb, l1, l2, … ln)
        if layers_to_keep is None:
            layers_to_keep = [len(h_states) - 1]   # last layer only

        # span of *current* sentence in this forward pass
        sent_start = len(ids_prefix) + len(ids_q) + len(ids_suffix) \
                   + (len(running_ids) - len(sent_ids))
        sent_end   = sent_start + len(sent_ids)

        pooled: Dict[str, List[float]] = {}
        for li, h in enumerate(h_states[1:], 1):   # skip embeddings
            if li not in layers_to_keep: continue
            pooled[f"layer_{li}"] = (
                _mean_pool(h, sent_start, sent_end)
                .to(torch.float16)
                .cpu()
                .numpy()
                .tolist()
            )

        out_rows.append(dict(
            sentence_id = sid,
            sentence    = sent,
            pooled_hs   = pooled,
        ))
    return out_rows


# ------------------------------ 4.  End-to-end capture ------------------

def run_probe_capture_for_categories(
    *,
    model,
    tok,
    cot_file: Path | str,
    selection_map: Dict[int, List[Tuple[int, str]]],
    layers: Optional[List[int]] = None,
    output_file: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Capture hidden-state vectors for the (question_id, sentence_id) pairs in
    `selection_map`.  Returns  {vectors, labels, attr_vecs}.
    """
    cot_data = {c["question_id"]: c["completion"]
                for c in json.loads(Path(cot_file).read_text("utf-8"))}

    vectors_by_layer: Dict[str, List[np.ndarray]] = {}
    labels: List[str] = []

    # running sums for attribute vectors
    sums: Dict[str, Dict[str, np.ndarray]] = {}
    counts: Dict[str, Dict[str, int]]      = {}

    for qid, sent_pairs in selection_map.items():
        if qid not in cot_data:
            logging.warning("CoT for question %d not found – skipping", qid)
            continue

        # use the raw question text (everything before first '<think>')
        cot_txt = cot_data[qid]
        q_end   = cot_txt.find("<think>")
        prompt_base = cot_txt[:q_end].strip() if q_end != -1 else cot_txt.strip()

        sentence_records = _run_sentence_level_inference(
            model,
            tok,
            prompt_base    = prompt_base,
            full_cot       = cot_txt,
            layers_to_keep = layers,
        )
        rec_lookup = {r["sentence_id"]: r for r in sentence_records}

        for sent_id, label in sent_pairs:
            rec = rec_lookup.get(sent_id)
            if not rec:
                logging.warning("Sentence %d not found in qid %d", sent_id, qid)
                continue
            for layer_name, vec in rec["pooled_hs"].items():
                vec_np = np.asarray(vec, dtype=np.float32)

                # dataset …
                vectors_by_layer.setdefault(layer_name, []).append(vec_np)

                # sums for attribute vectors …
                sums.setdefault(label, {}).setdefault(layer_name, 0.)
                counts.setdefault(label, {}).setdefault(layer_name, 0)
                sums[label][layer_name]   += vec_np
                counts[label][layer_name] += 1

            labels.append(label)

    # --- attribute vectors (category means) --------------------------------
    attr_vecs: Dict[str, Dict[str, torch.Tensor]] = {}
    for lbl in sums:
        attr_vecs[lbl] = {}
        for layer_name in sums[lbl]:
            attr_vecs[lbl][layer_name] = torch.tensor(
                sums[lbl][layer_name] / counts[lbl][layer_name])

    # --- optional disk dump -----------------------------------------------
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            "vectors": {k: [v.tolist() for v in vecs]
                        for k, vecs in vectors_by_layer.items()},
            "labels":  labels,
            "attr_vecs": {cat: {ln: v.numpy().tolist()
                                for ln, v in layerdict.items()}
                          for cat, layerdict in attr_vecs.items()},
        }
        out_path.write_text(json.dumps(serialisable), "utf-8")
        logging.info("Saved capture to %s", out_path)

    return {"vectors": vectors_by_layer, "labels": labels, "attr_vecs": attr_vecs}


# ------------------------------ 5.  Probing -----------------------------

def train_linear_probes(
    vectors_by_layer: Dict[str, List[np.ndarray]],
    labels: List[str],
    *,
    test_size: float    = 0.2,
    random_state: int   = 42,
) -> Tuple[Dict[str, LogisticRegression], Dict[str, Dict[str, float]]]:
    """
    Train one multinomial logistic-regression probe per layer.
    Returns (probes, metrics) where metrics[layer] = {'accuracy', 'f1'}.
    """
    probes:  Dict[str, LogisticRegression] = {}
    metrics: Dict[str, Dict[str, float]]   = {}

    y = np.array(labels)

    for layer_name, vecs in vectors_by_layer.items():
        X = np.vstack(vecs)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, stratify=y,
            test_size=test_size,
            random_state=random_state,
        )

        clf = LogisticRegression(
            max_iter   = 1000,
            multi_class= "multinomial",
            solver     = "saga",
            n_jobs     = 4,
        ).fit(X_tr, y_tr)

        y_pred  = clf.predict(X_te)
        acc     = accuracy_score(y_te, y_pred)
        f1      = f1_score(y_te, y_pred, average="weighted")

        probes[layer_name]  = clf
        metrics[layer_name] = {"accuracy": acc, "f1": f1}

    return probes, metrics


# ------------------------------ 6.  Utility -----------------------------

def save_attribute_vectors(
    attr_vecs: Dict[str, Dict[str, torch.Tensor]],
    out_dir: Path | str,
):
    """Write one file per category with its per-layer mean vectors (.pt)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat, layers in attr_vecs.items():
        torch.save({ln: v.cpu() for ln, v in layers.items()},
                   out_dir / f"{cat.replace(' ', '_')}.pt")
