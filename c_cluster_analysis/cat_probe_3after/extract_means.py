"""
extract_means.py
================
1. Runs *one* forward pass per sentence *including all layers*.
2. Maintains:
      • sum_pos[layer]  – Σ hidden for sentences in the category
      • sum_neg[layer]  – Σ hidden for all other sentences
      • n_pos[layer], n_neg[layer]  – counts
3. At the end writes
      means/layer_{i}.json =
        {"mean_pos": [...], "mean_neg": [...]}
Plus an (optional) probe-accuracy file if you ask for it.
"""
from collections import defaultdict
import json, logging, os, joblib
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from c_cluster_analysis.cat_probe_3after.probe_capture_general import run_batch_from_files

def extract_all_layers(
    model, tok, model_name,
    questions_file: str,
    full_cot_file: str,
    labels_file: str,
    out_dir: str,
    *,
    train_probe: bool = True,
    max_q: int | None = None,
):
    os.makedirs(f"{out_dir}/means", exist_ok=True)

    # 1-shot run that keeps **every** layer
    res = run_batch_from_files(
        model, tok,
        questions_file = questions_file,
        full_cot_file  = full_cot_file,
        layers_to_keep = None,          # <-- all layers
        max_questions  = max_q,
        output_file    = f"{out_dir}/hidden_all_layers.json",
    )

    labels_map = json.loads(Path(labels_file).read_text())
    nlayers = model.config.num_hidden_layers

    # ---- running totals ------------------------------------------
    sum_pos = [np.zeros(model.config.hidden_size) for _ in range(nlayers)]
    sum_neg = [np.zeros_like(sum_pos[0])            for _ in range(nlayers)]
    n_pos   = [0]*nlayers
    n_neg   = [0]*nlayers

    # ---- optional probe material ---------------------------------
    X_by_layer = [[] for _ in range(nlayers)]
    y_global   = []

    for entry in res:                                   # one question
        qid = str(entry["question_id"])
        sent_lbls = labels_map[qid]

        for srec, lbl in zip(entry["sentences"], sent_lbls):
            for li in range(1, nlayers+1):
                v = np.asarray(srec["pooled_hs"][f"layer_{li}"], dtype=np.float32)

                if lbl == 1:
                    sum_pos[li-1] += v;  n_pos[li-1] += 1
                else:
                    sum_neg[li-1] += v;  n_neg[li-1] += 1

                if train_probe:
                    X_by_layer[li-1].append(v)
            if train_probe:                 # same label for all layers
                y_global.append(lbl)

    # ------------- write means ------------------------------------
    for li in range(1, nlayers+1):
        mean_pos = (sum_pos[li-1]/max(n_pos[li-1],1)).tolist()
        mean_neg = (sum_neg[li-1]/max(n_neg[li-1],1)).tolist()
        Path(f"{out_dir}/means/layer_{li}.json").write_text(
            json.dumps({"mean_pos": mean_pos, "mean_neg": mean_neg})
        )

    # ------------- optional layer sweep probe ---------------------
    if train_probe:
        scoreboard = []
        for li in range(1, nlayers+1):
            X = np.vstack(X_by_layer[li-1])
            y = np.array(y_global)
            clf = LogisticRegression(max_iter=200, n_jobs=4).fit(X, y)
            acc = accuracy_score(y, clf.predict(X))
            joblib.dump(clf, f"{out_dir}/probes_layer_{li}.pth")
            scoreboard.append((li, acc))
        with open(f"{out_dir}/probe_scores.tsv","w") as fh:
            scoreboard.sort(key=lambda x: x[1], reverse=True)
            for li,acc in scoreboard:
                fh.write(f"{li}\t{acc:.4f}\n")
        best = scoreboard[0]
        logging.info("► best layer %d  acc=%.3f", *best)

if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--questions-file", required=True)
    ap.add_argument("--full-cot-file", required=True)
    ap.add_argument("--labels-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    extract_all_layers(
        model_path = args.model_path,
        questions_file = args.questions_file,
        full_cot_file  = args.full_cot_file,
        labels_file    = args.labels_file,
        out_dir        = args.out_dir,
        train_probe    = not args.no_probe,
        max_q          = args.max_questions,
    )
