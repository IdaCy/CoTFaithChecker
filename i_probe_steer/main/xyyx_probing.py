#!/usr/bin/env python3
"""
Distributed-ready version of the logistic-regression probe trainer.

*   Shards the *.pt activation files across all GPU ranks.
*   Uses HF Accelerate for easy multi-GPU launch.
*   Gathers activations back to rank-0, which trains & saves the probes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    nohup accelerate launch \
          --num_processes 4 \
          --mixed_precision no \
          i_probe_steer/main/xyyx_probing.py \
          > logs/probe_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
from __future__ import annotations
import os, json, glob, re, math, random, itertools, time
from pathlib import Path
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
import socket, os, sys
import sys, pathlib, os
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print("Working dir:", PROJECT_ROOT)

import torch
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # head-less
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc  # NEW
from sklearn.model_selection import train_test_split

from accelerate import Accelerator

import logging
LOG_FILE = "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(process)d - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w")]
)

print(f"on host {socket.gethostname()} (PID {os.getpid()}) ===",)
print("starting at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))


ANSWERS_DIR      = Path("e_confirm_xy_yx/outputs/matched_vals_gt")
ACTIVATIONS_DIR  = Path("h_hidden_space/outputs/f1_hint_xyyx/xyyx_deterministic")
PROBE_SAVE_DIR   = Path("linear_probes/max_sly_5k"); PROBE_SAVE_DIR.mkdir(exist_ok=True)

TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_PATH     = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

RANDOM_SEED            = 0
MAX_FILES              = None        # None = use every *_hidden.pt file
LOG_EVERY              = 50
MAX_SAMPLES_PER_LAYER  = 5_000       # subsample cap per layer (None = all)
PRINT_EVERY_LAYERS     = 5
SK_VERBOSE             = 0           # scikit-learn verbosity

# reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

accelerator   = Accelerator()
DEVICE        = accelerator.device
is_main       = accelerator.is_main_process

print(f"gather_object poly-fill; accelerator: {accelerator}")
if not hasattr(accelerator, "gather_object"):
    import torch.distributed as dist

    def _gather_object(obj):
        """Emulate accelerate.gather_object() on old versions."""
        if accelerator.num_processes == 1:
            return [obj]
        gathered = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(gathered, obj)
        return gathered

    accelerator.gather_object = _gather_object
print("launching at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

def p(*a, **kw):                      # rank-aware print
    accelerator.print(*a, **kw)

def iter_answer_files(answer_dir: Path):
    for fp in answer_dir.glob("*.json"):
        with open(fp) as f:
            yield json.load(f), fp.name

def parse_expected_from_fname(fname: str) -> str:
    m = re.search(r"_(?:gt|lt)_(YES|NO)_", fname)
    if not m:
        raise ValueError(f"Cannot parse YES/NO from {fname}")
    return m.group(1)

def hidden_files() -> list[Path]:
    paths = sorted(ACTIVATIONS_DIR.rglob("*_hidden.pt"))
    return paths if MAX_FILES is None else paths[:MAX_FILES]

def yes_no_logits(h: torch.Tensor,
                  yes_w: torch.Tensor, no_w: torch.Tensor,
                  yes_b: float,        no_b: float) -> tuple[float, float]:
    return (float(torch.dot(h, yes_w) + yes_b),
            float(torch.dot(h,  no_w) + no_b))

def yes_no_from_hidden(h: torch.Tensor,
                       yes_w: torch.Tensor, no_w: torch.Tensor,
                       yes_b: float,        no_b: float) -> str:
    s_yes, s_no = yes_no_logits(h, yes_w, no_w, yes_b, no_b)
    return "YES" if s_yes > s_no else "NO"

def softmax2(a: float, b: float) -> tuple[float, float]:
    mx = max(a, b)
    ea, eb = math.exp(a - mx), math.exp(b - mx)
    denom  = ea + eb
    return ea / denom, eb / denom

def main() -> None:
    print("loading answers at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    if is_main:
        a_counter, b_counter, same_counter = Counter(), Counter(), Counter()
        for data, _name in iter_answer_files(ANSWERS_DIR):
            questions = data["questions"] if isinstance(data, dict) else data
            for q in questions:
                a_counter[q["a_answers"][0]]   += 1
                b_counter[q["b_answers"][0]]   += 1
                same_counter[q["same"][0]]     += 1
        for k in ("YES", "NO"):
            a_counter.setdefault(k, 0)
            b_counter.setdefault(k, 0)
        for k in (True, False):
            same_counter.setdefault(k, 0)

        fig, ax = plt.subplots(figsize=(7, 4))
        groups   = ["A-answer", "B-answer", "Same"]
        yes_vals = [a_counter["YES"], b_counter["YES"], same_counter[True]]
        no_vals  = [a_counter["NO"],  b_counter["NO"],  same_counter[False]]
        x = np.arange(len(groups)); width = 0.35
        ax.bar(x - width/2, no_vals,  width, label="NO")
        ax.bar(x + width/2, yes_vals, width, label="YES / True")
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("# answers"); ax.set_title("Accumulated answers per category")
        ax.legend(); plt.tight_layout()
        plt.savefig(PROBE_SAVE_DIR / "answer_histogram.png")
        plt.close(fig)

    print("tokenizing1 at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

    with accelerator.main_process_first():
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        tok.pad_token = tok.eos_token

    ANSWER_TOKEN_IDS = {t: tok.encode(t, add_special_tokens=False)[0]
                        for t in ("YES", "NO")}
    if is_main:
        p(f"YES id={ANSWER_TOKEN_IDS['YES']}  |  NO id={ANSWER_TOKEN_IDS['NO']}")

    print("tokenizing2 at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

    p("Loading model to read out lm_head …")
    t0 = time.time()
    with accelerator.main_process_first():
        base_model = (AutoModelForCausalLM
                      .from_pretrained(MODEL_PATH,
                                       torch_dtype=torch.float32,
                                       low_cpu_mem_usage=True)
                      .to(DEVICE).eval())
    p(f"  ↳ done in {time.time()-t0:.1f}s")

    print("reading out lm_head at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

    W = base_model.lm_head.weight.detach().cpu()      # (vocab, h)
    b = (base_model.lm_head.bias.detach().cpu()
         if base_model.lm_head.bias is not None else None)
    yes_w = W[ANSWER_TOKEN_IDS["YES"]]
    no_w  = W[ANSWER_TOKEN_IDS["NO"]]
    yes_b = b[ANSWER_TOKEN_IDS["YES"]] if b is not None else 0.0
    no_b  = b[ANSWER_TOKEN_IDS["NO"]]  if b is not None else 0.0

    all_paths_total = hidden_files()
    all_paths       = all_paths_total[accelerator.process_index::
                                      accelerator.num_processes]
    if is_main:
        p(f"Processing {len(all_paths_total):,} files on "
          f"{accelerator.num_processes} ranks "
          f"({len(all_paths):,} / rank)")

    layer_buckets: dict[int, list[tuple[torch.Tensor, int]]] = defaultdict(list)

    iterator = (all_paths if not is_main
                else tqdm(all_paths, desc="hidden.pt", leave=False))

    print("iterating at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    start = time.time()
    for idx, hid_path in enumerate(iterator, 1):
        expected = parse_expected_from_fname(hid_path.name)      # YES / NO
        batch_hidden: list[torch.Tensor] = torch.load(hid_path)  # list[n_layers]

        last_layer = batch_hidden[-1]                # (B, H)
        preds = [yes_no_from_hidden(h.float(), yes_w, no_w, yes_b, no_b)
                 for h in last_layer]

        for L, layer_tensor in enumerate(batch_hidden):
            for h_vec, pred in zip(layer_tensor, preds):
                label = 1 if pred == expected else 0
                layer_buckets[L].append((h_vec.float(), label))

        if idx % LOG_EVERY == 0 and not is_main:
            p(f"rank {accelerator.process_index}: {idx}/{len(all_paths)} done")

    print("collecting at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    elapsed = time.time() - start
    p(f"Rank {accelerator.process_index}: collected "
      f"{sum(len(v) for v in layer_buckets.values()):,} samples "
      f"in {elapsed/60:.1f} min")

    accelerator.wait_for_everyone()
    bucket_shards = accelerator.gather_object(layer_buckets)  # list[dict] – 1/rank

    if not is_main:
        # no more work; free memory & exit
        del base_model, W, layer_buckets
        return

    print("merging at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    merged_buckets: dict[int, list[tuple[torch.Tensor, int]]] = defaultdict(list)
    for shard in bucket_shards:
        for L, pairs in shard.items():
            merged_buckets[L].extend(pairs)
    layer_buckets = merged_buckets
    p("All ranks gathered & merged")

    for L in sorted(layer_buckets)[:5]:
        p(f"  layer {L:2d} → {len(layer_buckets[L]):,} samples (showing first 5 layers)")

    print("training at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    results   = {}
    layer_ids = sorted(layer_buckets)

    p(f"\n=== Training probes for {len(layer_ids)} layers "
      f"(≤ {MAX_SAMPLES_PER_LAYER or 'all'} samples/layer) ===")

    print("training2 at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    bar = tqdm(layer_ids, desc="Layer", leave=True)
    t_global = time.time()

    print("entering loop at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    for i, L in enumerate(bar, 1):
        pairs = layer_buckets[L]
        if MAX_SAMPLES_PER_LAYER and len(pairs) > MAX_SAMPLES_PER_LAYER:
            pairs = random.sample(pairs, MAX_SAMPLES_PER_LAYER)

        X = torch.stack([p[0] for p in pairs]).numpy()
        y = np.array([p[1] for p in pairs])

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        t0 = time.time()
        probe = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=10_000,
            n_jobs=-1,
            verbose=SK_VERBOSE,
        ).fit(X_tr, y_tr)
        f1 = f1_score(y_val, probe.predict(X_val))
        dt = time.time() - t0

        results[L] = (f1, probe)
        bar.set_postfix(layer=L, valF1=f"{f1:.3f}", secs=f"{dt:.1f}")

        if i % PRINT_EVERY_LAYERS == 0:
            p(f"  ↳ layer {L:2d} finished | val-F1 {f1:.3f} | {dt:.1f}s")

    print("closing bar at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    bar.close()
    p(f"Total probe-training time: {(time.time()-t_global)/60:.1f} min")

    best_layer, (best_f1, best_probe) = max(results.items(), key=lambda kv: kv[1][0])
    p(f"\nBest layer = {best_layer}   (F1 = {best_f1:.3f})")

    outfile = PROBE_SAVE_DIR / f"linear_probe_layer{best_layer}.joblib"
    joblib.dump(best_probe, outfile)
    p(f"Probe saved → {outfile}")

    print("plotting at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
    layers = sorted(results); f1s = [results[L][0] for L in layers]
    plt.figure(figsize=(6,3))
    plt.plot(layers, f1s, marker="o")
    plt.xlabel("Layer #"); plt.ylabel("F1 on validation"); plt.grid(True)
    plt.title("Which layer encodes answer correctness?")
    plt.tight_layout()
    plt.savefig(PROBE_SAVE_DIR / "layer_f1_curve.png")
    p("Plot saved")

    sample_counts = [len(layer_buckets[L]) for L in layers]
    plt.figure(figsize=(6,3))
    plt.bar(layers, sample_counts)
    plt.xlabel("Layer #"); plt.ylabel("# samples")
    plt.title("Samples collected per layer")
    plt.tight_layout()
    plt.savefig(PROBE_SAVE_DIR / "layer_sample_counts.png")
    plt.close()
    p("layer_sample_counts.png saved")

    pairs_all = layer_buckets[best_layer]
    X_all = torch.stack([p[0] for p in pairs_all]).numpy()
    y_all = np.array([p[1] for p in pairs_all])
    y_pred = best_probe.predict(X_all)
    cm = confusion_matrix(y_all, y_pred)
    plt.figure(figsize=(3,3))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0,1], ["Wrong", "Correct"])
    plt.yticks([0,1], ["Wrong", "Correct"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    plt.title(f"Confusion matrix, layer {best_layer}")
    plt.tight_layout()
    plt.savefig(PROBE_SAVE_DIR / f"confusion_matrix_layer{best_layer}.png")
    plt.close()
    p("confusion_matrix saved")

    if hasattr(best_probe, "predict_proba"):
        probs = best_probe.predict_proba(X_all)[:,1]
    else:                                   # liblinear fallback
        probs = best_probe.decision_function(X_all)
    fpr, tpr, _ = roc_curve(y_all, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, marker=".")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={roc_auc:.3f}) layer {best_layer}")
    plt.tight_layout()
    plt.savefig(PROBE_SAVE_DIR / f"roc_layer{best_layer}.png")
    plt.close()
    p("ROC curve saved")

if __name__ == "__main__":
    main()
