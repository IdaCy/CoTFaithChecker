#!/usr/bin/env python
"""
Layer-sweep probe runner
========================
1. Runs Script-1’s capture once per *step* layers (default 5).
2. Immediately trains a logistic-regression probe for the
   chosen *category* and prints / logs the score.
3. Writes:
     • hidden/{layer_i}.json   – pooled vectors
     • probes/{layer_i}.pth    – scikit-learn joblib dump
     • sweep_results.tsv       – layer \t dev_acc

Assumes you already have
    – questions_file      (MMLU, etc.)
    – full_cot_file       (the generated CoT completions)
    – labels_file         mapping: {question_id: [sent_cat...]}

Only the thin wrapper is new; everything heavy re-uses the
`run_batch_from_files` function from Script-1.
"""
import json, joblib, logging, os, sys
from pathlib import Path
from typing import List, Dict

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- import the *original* machinery --------------------
from probe_capture_runner import (
    load_model_and_tokenizer,
    run_batch_from_files,
)

# --------------------------------------------------------
def _train_probe(vecs: List[List[float]],
                 labels: List[int]) -> (float, LogisticRegression):
    """tiny helper: train / eval 80-20 split, return acc and fitted clf"""
    X_train, X_dev, y_train, y_dev = train_test_split(
        vecs, labels, test_size=0.20, random_state=42, stratify=labels)
    clf = LogisticRegression(max_iter=200, n_jobs=4)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_dev, clf.predict(X_dev))
    return acc, clf


def layer_sweep(
    model_path: str,
    questions_file: str,
    full_cot_file: str,
    labels_file: str,                 # ⬅ category labels
    out_dir: str,
    *,
    sweep_step: int = 5,
    max_q: int | None = None,
):
    os.makedirs(f"{out_dir}/hidden", exist_ok=True)
    os.makedirs(f"{out_dir}/probes", exist_ok=True)

    model, tok, model_name, _ = load_model_and_tokenizer(model_path)
    num_layers = model.config.num_hidden_layers
    logging.info("model has %d transformer blocks", num_layers)

    labels_map: Dict[str, List[int]] = json.loads(Path(labels_file).read_text())

    sweep_log = []

    for li in range(1, num_layers + 1, sweep_step):
        logging.info("► capturing layer %d", li)

        hidden_json = f"{out_dir}/hidden/layer_{li}.json"
        res = run_batch_from_files(
            model, tok,
            questions_file = questions_file,
            full_cot_file  = full_cot_file,
            output_file    = hidden_json,
            layers_to_keep = [li],            # <-- key line
            max_questions  = max_q,
        )

        # ---- assemble vectors + labels ------------------
        vecs, lbls = [], []
        for entry in res:                      # one question
            qid = str(entry["question_id"])
            sent_lbls = labels_map[qid]        # list[int] same len as sentences
            for srec, l in zip(entry["sentences"], sent_lbls):
                vecs.append( srec["pooled_hs"][f"layer_{li}"] )
                lbls.append(l)

        acc, clf = _train_probe(vecs, lbls)
        joblib.dump(clf, f"{out_dir}/probes/layer_{li}.pth")
        sweep_log.append((li, acc))
        logging.info("layer %d  – dev-acc %.3f", li, acc)

    # ------------- summary file -------------------------
    sweep_log.sort(key=lambda x: x[1], reverse=True)
    with open(f"{out_dir}/sweep_results.tsv", "w") as fh:
        for li, acc in sweep_log:
            fh.write(f"{li}\t{acc:.4f}\n")

    best_layer, best_acc = sweep_log[0]
    logging.info("✓ best layer %d with acc %.3f", best_layer, best_acc)
    return best_layer


if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--questions-file", required=True)
    ap.add_argument("--full-cot-file", required=True)
    ap.add_argument("--labels-file", required=True,
                    help="JSON mapping {question_id: [sent_cat_int,…]}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sweep-step", type=int, default=5)
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    layer_sweep(
        model_path    = args.model_path,
        questions_file= args.questions_file,
        full_cot_file = args.full_cot_file,
        labels_file   = args.labels_file,
        out_dir       = args.out_dir,
        sweep_step    = args.sweep_step,
        max_q         = args.max_questions,
    )
