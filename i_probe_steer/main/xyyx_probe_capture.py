#!/usr/bin/env python3
"""
Linear-probe trainer for the XY⇄YX faithfulness experiment.

Targets:
Targets:
    y = 1 … model answered correctly *and* gave different answers to the two variants
    y = 0 … model gave the same answer to both variants and that answer was wrong

All bookkeeping comes from the answer JSONs – we never peek at `lm_head`.

CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup accelerate launch \
      --num_processes 4 \
      --mixed_precision no \
      i_probe_steer/main/xyyx_probe_capture.py \
      > logs/launcher_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""


from __future__ import annotations
import os, re, json, math, random, time, socket, pathlib
from pathlib import Path
from collections import defaultdict, Counter
from zoneinfo import ZoneInfo

import sys, pathlib, os, logging
from pathlib import Path

import logging
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
import socket, os, sys
from accelerate import Accelerator

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print("Working dir:", PROJECT_ROOT)

LOG_FILE = "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(process)d - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w")]
)

accelerator = Accelerator()
if accelerator.is_main_process:
    logging.getLogger().addHandler(logging.StreamHandler())

print(f"on host {socket.gethostname()} (PID {os.getpid()}) ===",)
print("starting at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))


import torch, joblib
import numpy as np
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

from accelerate import Accelerator

print("starting at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))


# CONFIG
# ────────────────────────────────────────────────────────────────────────────
ANSWERS_DIRS      = [
    Path("e_confirm_xy_yx/outputs/matched_vals_gt"),
    Path("e_confirm_xy_yx/outputs/matched_vals_lt"),
]
ACTIVATIONS_DIR   = Path("h_hidden_space/outputs/f1_hint_xyyx/xyyx_deterministic/gt_lt_completions_1")
#QUESTION_JSON_DIR = Path("data/chainscope/questions_json/linked")
QUESTION_JSON_ROOT = Path("data/chainscope/questions_json/linked")

PROBE_SAVE_DIR    = Path("linear_probes/realyesno_None4k")  # results end up here
PROBE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

INFERENCE_BATCH_SIZE   = 32   # must match the batch size used in `run_inference`
MAX_SAMPLES_PER_LAYER  = 4_000
PRINT_EVERY_LAYERS     = 5
RANDOM_SEED            = 0
MAX_FILES = None
# ────────────────────────────────────────────────────────────────────────────

print("settings at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

acc       = Accelerator()
DEVICE    = acc.device
is_main   = acc.is_main_process
p         = acc.print                                                    # rank-aware print
_now      = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print("accelerator setup at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
# ─── make accelerate≤0.20 look like ≥0.21 ─────────────────────────
if not hasattr(acc, "gather_object"):
    import torch.distributed as dist

    def _gather_object(obj):
        """
        Minimal stand-in for accelerate.gather_object().
        Works for any picklable Python object.
        """
        if acc.num_processes == 1:
            return [obj]                      # nothing to gather
        gathered = [None] * acc.num_processes
        dist.all_gather_object(gathered, obj) # inplace
        return gathered

    acc.gather_object = _gather_object
# ──────────────────────────────────────────────────────────────────

print("building answers map at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
# │ 1.  build a QUESTION-ID  →  {expected, actual, same} lookup
answers_map: dict[int, dict[str, object]] = {}

def load_answers():
    for dir_ in ANSWERS_DIRS:
        for fp in dir_.glob("*.json"):
            with open(fp) as f:
                raw = json.load(f)
            questions = raw["questions"] if isinstance(raw, dict) else raw
            for q in questions:
                same_flag = q["same"][0] if isinstance(q["same"], list) else q["same"]
                q_id_no   = q["question_id"]
                q_id_yes  = q["question_yes_id"]
                answers_map[q_id_no]  = {"expected": "NO",  "actual": q["a_answers"][0],
                                         "same": same_flag}
                answers_map[q_id_yes] = {"expected": "YES", "actual": q["b_answers"][0],
                                         "same": same_flag}

print("loading answer files at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
if is_main:
    print(f"[{_now()}] loading answer files …")
load_answers()
acc.wait_for_everyone()

# optional quick sanity print
if is_main:
    cnt = Counter((v["expected"], v["actual"]) for v in answers_map.values())
    print("answers_map built:",
          ", ".join([f"{k}:{v:,}" for k,v in cnt.items()]))
    
print("DONE ! at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 2.  utilities to map a hidden-file row → question-id                    │
# ╰──────────────────────────────────────────────────────────────────────────╯
_DATASET_CACHE: dict[str, list[dict]] = {}

_STEM2PATH: dict[str, Path] = {}

def _build_index_once():
    if _STEM2PATH:                    # already built
        return
    for p in QUESTION_JSON_ROOT.rglob("*.json"):
        _STEM2PATH[p.stem] = p
    if not _STEM2PATH:
        raise RuntimeError(f"No *.json found under {QUESTION_JSON_ROOT}")

def get_dataset_questions(stem: str) -> list[dict]:
    _build_index_once()
    fp = _STEM2PATH.get(stem)
    if fp is None:
        raise FileNotFoundError(f"Could not find {stem}.json under {QUESTION_JSON_ROOT}")
    with open(fp) as f:
        raw = json.load(f)
    return raw["questions"] if isinstance(raw, dict) else raw


_BATCH_RE = re.compile(r"_batch(\d+)_hidden\.pt$")

def question_ids_for_hidden_file(hid_path: Path, batch_size: int, actual_batch_len: int) -> list[int]:
    """
    Returns the list of question-ids (length = actual_batch_len) that the hidden
    tensor rows correspond to.
    """
    m = _BATCH_RE.search(hid_path.name)
    if m is None:
        raise ValueError(f"Bad hidden filename: {hid_path.name}")
    batch_idx  = int(m.group(1))
    dataset_stem = hid_path.name[:m.start()]           # strip “…_batchX_hidden.pt”
    q_list     = get_dataset_questions(dataset_stem)
    start      = batch_idx * batch_size
    return [q_list[start + j]["question_id"]           # key name in those JSONs
            for j in range(actual_batch_len)]

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 3.  collect (h_vec, label) pairs for every layer                         │
# ╰──────────────────────────────────────────────────────────────────────────╯
layer_buckets: dict[int, list[tuple[torch.Tensor, int]]] = defaultdict(list)

all_hidden_paths = sorted(ACTIVATIONS_DIR.rglob("*_hidden.pt"))
#   shard the work across accelerator ranks
hidden_paths = all_hidden_paths[acc.process_index::acc.num_processes]

if is_main:
    print(f"[{_now()}] • {len(all_hidden_paths):,} hidden files total")
    print(f"[{_now()}] • {len(hidden_paths):,} files on rank-{acc.process_index}")

progress = tqdm(hidden_paths, desc="collect", disable=not is_main)
for hid_fp in progress:
    batch_hidden: list[torch.Tensor] = torch.load(hid_fp)   # list[n_layers]  each (B,H)
    batch_len     = batch_hidden[0].size(0)
    q_ids         = question_ids_for_hidden_file(hid_fp, INFERENCE_BATCH_SIZE, batch_len)

    # outer loop: row-in-batch
    for row_idx, q_id in enumerate(q_ids):
        meta = answers_map.get(q_id)
        if meta is None:                 # happens if you didn't copy both gt & lt directories
            continue

        #correct = (meta["actual"] == meta["expected"])
        #if not correct and not meta["same"]:
        #    continue                     # skip “ordinary” mistakes – we only want same-answer errors

        #label = 1 if correct else 0

        correct = (meta["actual"] == meta["expected"])   # True ↔ answer matches ground truth
        same    = meta["same"]                           # True ↔ model answered both variants identically
        # keep only:
        #   - correct & not-same   → “faithful-correct”   (label 1)
        #   - wrong   &     same   → “unfaithful-wrong”   (label 0)
        # drop everything else (right-but-same, wrong-but-different)
        if correct == same:          # (True,True) or (False,False) → skip
            continue

        label = 1 if correct else 0  # 1 = faithful-correct, 0 = unfaithful-wrong

        for L, layer_tensor in enumerate(batch_hidden):
            layer_buckets[L].append((layer_tensor[row_idx].float(), label))

progress.close()
acc.wait_for_everyone()

bucket_shards = acc.gather_object(layer_buckets)   # list[dict] – 1 per rank

if not is_main:
    # free some memory and exit early
    del layer_buckets
    sys.exit(0) 

# merge rank shards
merged: dict[int, list[tuple[torch.Tensor,int]]] = defaultdict(list)
for shard in bucket_shards:
    for L, pairs in shard.items():
        merged[L].extend(pairs)
layer_buckets = merged

print(f"[{_now()}] gathered {sum(len(v) for v in layer_buckets.values()):,} labelled samples")

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 4.  layer-wise logistic-regression probes                                │
# ╰──────────────────────────────────────────────────────────────────────────╯
results = {}
t_global = time.time()
for L in tqdm(sorted(layer_buckets), desc="train probes"):
    pairs = layer_buckets[L]
    if MAX_SAMPLES_PER_LAYER and len(pairs) > MAX_SAMPLES_PER_LAYER:
        pairs = random.sample(pairs, MAX_SAMPLES_PER_LAYER)

    X = torch.stack([p[0] for p in pairs]).numpy()
    y = np.array([p[1] for p in pairs])

    if len(np.unique(y)) < 2:            # all-positive or all-negative – skip
        continue

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    probe = LogisticRegression(
        penalty="l2", solver="saga",
        max_iter=10_000, n_jobs=-1, verbose=0
    ).fit(X_tr, y_tr)
    f1 = f1_score(y_val, probe.predict(X_val))
    results[L] = (f1, probe)

"""
best_layer, (best_f1, best_probe) = max(results.items(), key=lambda kv: kv[1][0])
out_path = PROBE_SAVE_DIR / f"linear_probe_layer{best_layer}.joblib"
joblib.dump(best_probe, out_path)
print(f"[{_now()}] best layer = {best_layer}  (val-F1 {best_f1:.3f})  → saved →  {out_path}")
"""
layers_sorted = sorted(results)
num_layers    = len(layers_sorted)

# 1.  best overall
best_layer_overall, (best_f1_overall, best_probe_overall) = \
    max(results.items(), key=lambda kv: kv[1][0])

# 2.  best of the second half
second_half   = [L for L in layers_sorted if L >= num_layers // 2]
best_layer_2ndhalf = max(second_half, key=lambda L: results[L][0])
best_f1_2ndhalf, best_probe_2ndhalf = results[best_layer_2ndhalf]

# 3.  penultimate layer (-2) – guard against tiny models
penult_layer = layers_sorted[-2] if num_layers >= 2 else layers_sorted[-1]
penult_f1,   penult_probe   = results[penult_layer]

# 4.  best of the last few layers
LAST_FEW          = 4                                   # tweak if you like
lastfew_layers    = layers_sorted[-LAST_FEW:]
best_layer_lastfew = max(lastfew_layers, key=lambda L: results[L][0])
best_f1_lastfew,  best_probe_lastfew = results[best_layer_lastfew]

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 5.  quick diagnostics                                                    │
# ╰──────────────────────────────────────────────────────────────────────────╯
layers = sorted(results)
f1s    = [results[L][0] for L in layers]
plt.figure(figsize=(6,3))
plt.plot(layers, f1s, marker="o"); plt.grid(True)
plt.title("Layer-wise validation F1"); plt.xlabel("layer"); plt.ylabel("F1")
plt.tight_layout(); plt.savefig(PROBE_SAVE_DIR / "layer_f1_curve.png"); plt.close()

sample_counts = [len(layer_buckets[L]) for L in layers]
plt.figure(figsize=(6,3))
plt.bar(layers, sample_counts)
plt.title("#samples per layer"); plt.xlabel("layer"); plt.ylabel("count")
plt.tight_layout(); plt.savefig(PROBE_SAVE_DIR / "layer_sample_counts.png"); plt.close()

# confusion & ROC for best layer
pairs_all = layer_buckets[best_layer]
X_all     = torch.stack([p[0] for p in pairs_all]).numpy()
y_all     = np.array([p[1] for p in pairs_all])
y_pred    = best_probe.predict(X_all)

plt.figure(figsize=(3,3))
cm = confusion_matrix(y_all, y_pred)
plt.imshow(cm, cmap="Blues")
for (i,j),v in np.ndenumerate(cm):
    plt.text(j,i,f"{v:,}",ha="center",va="center")
plt.xticks([0,1],["wrong","correct"]); plt.yticks([0,1],["wrong","correct"])
plt.tight_layout(); plt.savefig(PROBE_SAVE_DIR / f"confusion_matrix_layer{best_layer}.png"); plt.close()

probs = (best_probe.predict_proba(X_all)[:,1]
         if hasattr(best_probe,"predict_proba")
         else best_probe.decision_function(X_all))
fpr,tpr,_ = roc_curve(y_all, probs); roc_auc = auc(fpr,tpr)
plt.figure(figsize=(4,4))
plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--")
plt.title(f"ROC AUC {roc_auc:.3f}  (layer {best_layer})")
plt.tight_layout(); plt.savefig(PROBE_SAVE_DIR / f"roc_layer{best_layer}.png"); plt.close()

print(f"[{_now()}] all done – diagnostics written to {PROBE_SAVE_DIR}")

