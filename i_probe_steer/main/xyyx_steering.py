"""
Uses the trained linear probe to *steer* Llama-8B on MMLU questions:
for each option (A/B/…) we run a single forward pass, feed the hidden vector of
layer <best> into the probe, and pick the option whose probe score is highest.
"""

# %% ------------------------- imports --------------------------
from __future__ import annotations
import json, os, re, math, random, time, logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

# helper from your earlier script
def construct_prompt(data_entry: Dict) -> str:
    question = data_entry["question"]
    options  = "\n".join([f"[ {key} ] {value}"
                          for key, value in data_entry.items()
                          if key in list("ABCDEFGHIJK")])
    hint_text = data_entry.get("hint_text")
    prompt = (f"Question:\n\n{question}\n\nOptions:\n\n{options}"
              "\n\nPlease answer with the letter of the corresponding to the correct option.")
    if hint_text:
        prompt = f"{prompt}\n\n{hint_text}"
    return prompt

# %% -------------------- configuration -------------------------
MODEL_PATH  = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PROBE_DIR   = Path("linear_probes")      # where *.joblib is stored
DATA_DIR    = Path("data/mmlu")
QUESTION_FILE = DATA_DIR / "input_mcq_data.json"
HINT_FILE     = DATA_DIR / "hints_sycophancy.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# %% ---------------- load model & tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    output_hidden_states=True           # <- so we get per-layer activations
).to(DEVICE).eval()
hidden_size = model.config.hidden_size
logging.info("Model & tokenizer loaded")

# %% ------------- load best probe (auto-detect file) -----------
probes = sorted(PROBE_DIR.glob("linear_probe_layer*.joblib"))
if not probes:
    raise FileNotFoundError("No probe .joblib found in linear_probes/")
probe_path = probes[0]          # there is only one; else pick max F1 manually
best_layer = int(re.search(r"layer(\d+)", probe_path.stem).group(1))
probe      = joblib.load(probe_path)
w_probe    = torch.tensor(probe.coef_[0], dtype=torch.float32)
b_probe    = float(probe.intercept_[0])
logging.info(f"Loaded probe from {probe_path} (layer {best_layer})")

def probe_score(h: torch.Tensor) -> float:
    """Return sigmoid score that the answer will be correct."""
    z = torch.dot(h, w_probe.to(h)) + b_probe
    return torch.sigmoid(z).item()

# %% --------------------- data loading -------------------------
with open(QUESTION_FILE) as f:
    questions = json.load(f)
with open(HINT_FILE) as f:
    hints = {h["question_id"]: h for h in json.load(f)}

for q in questions:
    h = hints.get(q["question_id"])
    if h:       # add hint text so that construct_prompt() will inject it
        q["hint_text"] = h["hint_text"]

logging.info(f"Loaded {len(questions):,} questions")

# %% --------------- steering-time helper functions -------------
LETTERS = list("ABCDEFGHIJK")

def token_ids_for_letter(letter: str) -> List[int]:
    """Return the token id(s) that encode a letter with leading space,
    e.g. ' D'."""
    return tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]

def score_option(prompt_ids: torch.Tensor,
                 option_letter: str) -> float:
    """
    Forward pass with the prompt + letter token.  Return probe score
    on the hidden of that letter at layer <best_layer>.
    """
    answer_ids = token_ids_for_letter(option_letter)
    if len(answer_ids) != 1:
        raise ValueError(f"Letter {option_letter} splits into >1 tokens")
    ids = torch.cat([prompt_ids, torch.tensor(answer_ids, device=prompt_ids.device)])
    out = model(ids.unsqueeze(0))
    h_vec = out.hidden_states[best_layer][0, -1]     # last token of layer k
    return probe_score(h_vec.float().cpu())

def steer_answer(question_entry: Dict) -> str:
    prompt = construct_prompt(question_entry)
    prompt_ids = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False)["input_ids"][0].to(DEVICE)
    # collect available letters in this entry (A–D for most MMLU)
    letters = [L for L in LETTERS if L in question_entry]
    scores  = {L: score_option(prompt_ids, L) for L in letters}
    # pick highest-scoring letter
    return max(scores.items(), key=lambda kv: kv[1])[0]

# %% -------------------- run steering loop ---------------------
preds, golds = [], []
all_scores   = []          # for ROC
for q in tqdm(questions, desc="Steering"):
    gold = q["correct"]
    pred = steer_answer(q)
    preds.append(pred)
    golds.append(gold)
    # probe score of the *predicted* answer for ROC analysis
    prompt_ids = tokenizer(construct_prompt(q),
                           add_special_tokens=False,
                           return_tensors="pt")["input_ids"][0].to(DEVICE)
    all_scores.append(score_option(prompt_ids, pred))

acc = accuracy_score(golds, preds)
logging.info(f"Steered accuracy = {acc*100:.2f}% ({sum(np.array(golds)==np.array(preds))}/{len(golds)})")

print(classification_report(golds, preds, digits=3))

# %% ----------------------- graphics ---------------------------
cm = confusion_matrix(golds, preds, labels=LETTERS[:4])
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(4), LETTERS[:4]); plt.yticks(range(4), LETTERS[:4])
plt.xlabel("Predicted"); plt.ylabel("True")
for (i,j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.title("Confusion matrix (steered)")
plt.tight_layout()
plt.savefig("confusion_matrix_steered.png")
plt.close()
logging.info("Saved confusion_matrix_steered.png")

# ROC (treat “correct” vs “incorrect” as binary)
y_true = [(p==g) for p,g in zip(preds, golds)]
fpr, tpr, _ = roc_curve(y_true, all_scores)
roc_auc     = auc(fpr, tpr)
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Probe ROC on steered answers")
plt.legend(); plt.tight_layout()
plt.savefig("roc_steered.png"); plt.close()
logging.info("Saved roc_steered.png")

# %% ------------------- optional: save preds -------------------
out_path = Path("steered_preds.json")
with open(out_path,"w") as f:
    json.dump([{"question_id": q["question_id"],
                "gold": g, "pred": p} for q,g,p in zip(questions,golds,preds)],
              f, indent=2)
logging.info(f"Wrote per-question predictions → {out_path}")
