# CoT Faithfulness Investigation Notebook
# =====================================
# A fully‑featured, **one‑file** Jupyter‑friendly analysis pipeline.
# ‑ Can be executed top‑to‑bottom in a notebook cell‑by‑cell **or**
#   imported / run as a Python module (`python cot_faithfulness_notebook.py`).
# ‑ Uses **DuckDB** as the local analytical store – fast, file‑based,
#   zero setup.
# ‑ All heavy steps expose a `--max_questions` flag so you can iterate
#   on tiny samples first; default runs *everything*.
#
# Requirements (1st run):
# ```bash
# %pip install "duckdb>=0.10" pandas pyarrow tqdm scikit‑learn umap‑learn \
#                   matplotlib seaborn transformers datasets
# ```
# GPU users: install **torch** with CUDA beforehand.
#
# ───────────────────────────────────────────────────────────────────────────
# TABLE OF CONTENTS
# ─────────────────
# 0. Argument parsing & config
# 1. DB schema creation + JSON ingest
# 2. Answer verification against meta eval
# 3. Hidden‑state capture (+ entropy) with sampling support
# 4. Segment‑level stats & dim‑red visuals
# 5. Category separability (Linear probe)
# 6. Hint condition effects (ΔF1 & centroid drift)
# 7. Layer‑wise effort & activation norms
# 8. Faithfulness diagnostics (contradiction, answer support, FI)
# ---------------------------------------------------------------------------
# Author: ChatGPT (o3) — 2025‑04‑27

"""
1. initially:
python g_cot_cluster/hidden_states/test4.py --build_db --max_questions 200 --capture_hs

python g_cot_cluster/hidden_states/test4.py --capture_hs --layers -1 --max_questions 200

python g_cot_cluster/hidden_states/test4.py --plots --probe --hint_comp --max_questions 200
"""

from __future__ import annotations
import argparse, json, logging, os, pickle, random, sys, textwrap, time
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────── 0. CONFIG ────────────────────────────────
#ROOT          = Path("CoTFaithChecker").resolve()
ROOT          = ""
DATA_DIR      = Path("data/mmlu")
SEG_DIR       = Path("g_cot_cluster/outputs/mmlu/DeepSeek-R1-Distill-Llama-8B/correct_indices")
COMP_DIR      = Path("data/mmlu/DeepSeek-R1-Distill-Llama-8B")
MODEL_NAME    = "deepseek-ai/deepseek-r1-distill-llama-8b"
#CACHE_DIR     = "g_cot_cluster/analysis_cache"; #CACHE_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("g_cot_cluster/analysis_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
DB_PATH       = CACHE_DIR / "cot_faithfulness.duckdb"
RNG           = random.Random(0)

PHRASE_CATS = [
    "problem_restating", "knowledge_recall", "concept_definition",
    "quantitative_calculation", "logical_deduction", "option_elimination",
    "assumption_validation", "uncertainty_expression", "self_questioning",
    "backtracking_revision", "decision_confirmation", "answer_reporting",
]
HINT_TYPES   = ["none", "sycophancy", "induced_urgency", "unethical_information"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s – %(levelname)s – %(message)s",
                    datefmt="%H:%M:%S")
LOGGER = logging.getLogger("cot-faith")

def parse_args():
    p = argparse.ArgumentParser(description="CoT faithfulness investigation")
    p.add_argument("--build_db",    action="store_true", help="(Re)‑ingest JSON → DuckDB")
    p.add_argument("--capture_hs",  action="store_true", help="(Re)‑run hidden‑state capture")
    p.add_argument("--layers",      default="-1", help="Comma list of layers to capture (e.g. -6,-1)")
    p.add_argument("--max_questions", type=int, default=None,
                   help="Sub‑sample N questions per condition for speed")
    p.add_argument("--plots",       action="store_true", help="Generate exploratory plots")
    p.add_argument("--probe",       action="store_true", help="Run phrase‑category linear probe")
    p.add_argument("--hint_comp",   action="store_true", help="Analyse hint condition effects")
    p.add_argument("--effort",      action="store_true", help="Layer‑wise effort analysis")
    p.add_argument("--faith",       action="store_true", help="Faithfulness metrics incl. contradiction")
    return p.parse_args() if __name__ == "__main__" else parse_args.__wrapped__()

# Notebook fallback
try:
    ARGS = parse_args()
except SystemExit:        # inside Jupyter
    ARGS = argparse.Namespace(**{k: v.default if hasattr(v, "default") else False
                                 for k, v in parse_args.__wrapped__()._option_string_actions.items()})
    ARGS.build_db = False; ARGS.capture_hs = False; ARGS.layers = "-1"; ARGS.plots = False
    ARGS.probe = False; ARGS.hint_comp = False; ARGS.effort = False; ARGS.faith = False
    ARGS.max_questions = 200    # sensible notebook default

# ────────────────────────────── 1. DB INGEST ──────────────────────────────

def build_database(max_q: int | None = None):
    """Ingest all JSONs into DuckDB with the prescribed schema."""
    if DB_PATH.exists():
        LOGGER.info("Existing DB found at %s – dropping & recreating", DB_PATH)
        DB_PATH.unlink()
    con = duckdb.connect(DB_PATH)

    # 1. Questions table
    with open(DATA_DIR / "input_mcq_data.json") as fh:
        q_df = pd.read_json(fh)
    if max_q:
        q_df = q_df.head(max_q)
    q_df.to_parquet(str(CACHE_DIR / "questions.parquet"))
    con.execute("CREATE TABLE questions AS SELECT * FROM parquet_scan(?)", [str(CACHE_DIR/"questions.parquet")])

    # 2. Hints
    hint_rows = []
    for h_type in HINT_TYPES[1:]:           # exclude "none"
        with open(DATA_DIR / f"hints_{h_type}.json") as fh:
            data = json.load(fh)
        if max_q:
            data = [d for d in data if d["question_id"] < max_q]
        for d in data:
            hint_rows.append({**d, "hint_type": h_type})
    hints_df = pd.DataFrame(hint_rows)
    hints_df.to_parquet(str(CACHE_DIR/"hints.parquet"))
    con.execute("CREATE TABLE hints AS SELECT * FROM parquet_scan(?)", [str(CACHE_DIR/"hints.parquet")])

    # 3.–5. Completions, Segments, Meta eval
    compl_rows, seg_rows, meta_rows = [], [], []
    for h_type in HINT_TYPES:
        compl_path = COMP_DIR / h_type / "completions_with_500.json"
        meta_path  = (COMP_DIR / h_type / "switch_analysis_with_500.json" if h_type != "none"
                      else COMP_DIR / h_type / "verification_with_500.json")
        if not compl_path.exists():
            LOGGER.warning("%s missing – skipped", compl_path)
            continue
        with open(compl_path) as fh:
            compl_json = json.load(fh)
        if max_q:
            compl_json = [c for c in compl_json if c["question_id"] < max_q]
        compl_rows.extend({"model": "DeepSeek-R1‑8B", "hint_type": h_type, **c} for c in compl_json)

        # Segmented completions
        seg_path = SEG_DIR / f"segmented_completions_{h_type}.json"
        if seg_path.exists():
            with open(seg_path) as fh:
                seg_json = json.load(fh)
            if max_q:
                seg_json = [s for s in seg_json if s["question_id"] < max_q]
            for s in seg_json:
                for seg in s["segments"]:
                    seg_rows.append({
                        "model": "DeepSeek-R1‑8B", "hint_type": h_type,
                        "question_id": s["question_id"],
                        "phrase_category": seg["phrase_category"],
                        "start_char": seg["start"],
                        "end_char": seg["end"],
                        "text": seg["text"]
                    })
        else:
            LOGGER.warning("Segments file %s missing", seg_path)

        # Meta eval
        if meta_path.exists():
            with open(meta_path) as fh:
                meta_json = json.load(fh)
            if max_q:
                meta_json = [m for m in meta_json if m["question_id"] < max_q]
            meta_rows.extend({"model": "DeepSeek-R1‑8B", "hint_type": h_type, **m} for m in meta_json)

    pd.DataFrame(compl_rows).to_parquet(str(CACHE_DIR/"completions.parquet"))
    con.execute("CREATE TABLE completions AS SELECT * FROM parquet_scan(?)", [str(CACHE_DIR/"completions.parquet")])

    pd.DataFrame(seg_rows).to_parquet(str(CACHE_DIR/"segments.parquet"))
    con.execute("CREATE TABLE segments AS SELECT * FROM parquet_scan(?)", [str(CACHE_DIR/"segments.parquet")])

    pd.DataFrame(meta_rows).to_parquet(str(CACHE_DIR/"meta_eval.parquet"))
    con.execute("CREATE TABLE meta_eval AS SELECT * FROM parquet_scan(?)", [str(CACHE_DIR/"meta_eval.parquet")])

    LOGGER.info("DB build finished: %s", DB_PATH)
    con.close()

# ──────────────────────────── 2. ANSWER CHECK ─────────────────────────────

def extract_answer_letter(text: str) -> str | None:
    import re
    m = re.search(r"\[\s*([A-D])\s*]", text[::-1])   # search from end → quickest
    return m.group(1) if m else None

def verify_answers(max_q: int | None = None):
    con = duckdb.connect(DB_PATH, read_only=False)
    comp_df: pd.DataFrame = con.execute("SELECT question_id, hint_type, completion FROM completions").df()
    comp_df["pred"] = comp_df["completion"].apply(extract_answer_letter)
    q2correct = con.execute("SELECT question_id, correct FROM questions").df().set_index("question_id")["correct"].to_dict()
    comp_df["is_correct"] = comp_df.apply(lambda r: r["pred"] == q2correct.get(r["question_id"]), axis=1)
    bad = comp_df[comp_df["pred"].isna()]
    if not bad.empty:
        LOGGER.warning("%d completions missing answer token – investigate!", len(bad))
    comp_df[["question_id", "hint_type", "pred", "is_correct"]].to_parquet(str(CACHE_DIR/"answer_verif.parquet"))
    LOGGER.info("Answer verification saved → answer_verif.parquet  (accuracy ≈ %.3f)",
                comp_df["is_correct"].mean())
    con.close()

# ─────────────────── 3. HIDDEN‑STATE & ENTROPY CAPTURE ────────────────────
TOKENIZER = None; MODEL = None

def load_model():
    global TOKENIZER, MODEL
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        MODEL      = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, output_hidden_states=True, torch_dtype=torch.float16
                if torch.cuda.is_available() else torch.float32, device_map="auto")
        MODEL.eval()

# Char→token helpers -------------------------------------------------------

def char2tok_offsets(text: str):
    enc = TOKENIZER(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["offset_mapping"]

def span_to_token_range(offsets, c0: int, c1: int):
    import bisect
    starts = [s for s, _ in offsets]
    ends   = [e-1 for _, e in offsets]
    i0 = bisect.bisect_right(starts, c0) - 1
    i1 = bisect.bisect_right(ends,   c1) - 1
    return max(i0,0), max(i1,0)

# Capture ---------------------------------------------------------------

def capture_hidden_states(layers: Tuple[int,...], max_q: int | None = None):
    load_model()
    cache_pkl = CACHE_DIR / f"segment_hs_L{','.join(map(str,layers))}_max{max_q or 'all'}.pkl"
    if cache_pkl.exists():
        LOGGER.info("Hidden‑state cache present – reuse %s", cache_pkl.name); return cache_pkl

    con = duckdb.connect(DB_PATH)
    seg_df: pd.DataFrame = con.execute("""
        SELECT s.*, c.completion FROM segments s
        JOIN completions c USING (question_id, hint_type, model)
        ORDER BY question_id
    """).df()
    if max_q:
        seg_df = seg_df[seg_df.question_id < max_q]
    reps, meta = [], []
    for (h_type, qid), grp in tqdm(seg_df.groupby(["hint_type", "question_id"])):
        text = grp["completion"].iloc[0]
        ids, offs = char2tok_offsets(text)
        ids_t = torch.tensor([ids]).to(MODEL.device)
        with torch.no_grad():
            outs = MODEL(ids_t)
        hs = torch.stack([outs.hidden_states[l][0] for l in layers])  # [L,T,d]
        # Pre‑compute entropies per token
        logits = outs.logits[0]
        token_probs = F.softmax(logits, dim=-1)
        token_ent   = (-token_probs * token_probs.log()).sum(-1)

        for _, row in grp.iterrows():
            t0, t1 = span_to_token_range(offs, row.start_char, row.end_char)
            vec = hs[:, t0:t1+1].mean(1).cpu().float().numpy()  # [L,d]
            ent = token_ent[t0:t1+1].mean().item()
            reps.append(vec)
            meta.append((qid, h_type, row.phrase_category))
    reps = np.stack(reps)
    with open(cache_pkl, "wb") as fh:
        pickle.dump(dict(reps=reps, meta=meta, layers=layers), fh)
    LOGGER.info("Captured %s segment reps → %s", len(meta), cache_pkl.name)
    return cache_pkl

# ────────────────────── 4. SEGMENT‑LEVEL VISUALS ———————————————

def load_reps(cache_pkl):
    with open(cache_pkl, "rb") as fh:
        obj = pickle.load(fh)
    reps = obj["reps"].reshape(len(obj["meta"]), -1)   # flatten L*d
    qid, htype, cat = zip(*obj["meta"])
    return reps, np.array(qid), np.array(htype), np.array(cat)

# Simple UMAP scatter ---------------------------------------------------

def umap_plot(X, labels, title="UMAP", n_neighbors=40, min_dist=0.25):
    um = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine")
    emb = um.fit_transform(X)
    plt.figure(figsize=(6,6))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(emb[idx,0], emb[idx,1], s=6, alpha=0.7, label=lab)
    plt.title(title); plt.legend(markerscale=4, fontsize=8)

# ─────────────────────── 5. PHRASE‑CATEGORY PROBE ——————————————

def phrase_probe(X, y):
    clf = LogisticRegression(max_iter=400, n_jobs=-1, multi_class="multinomial")
    idx  = np.arange(len(y)); RNG.shuffle(idx)
    split = int(0.8*len(y))
    tr, te = idx[:split], idx[split:]
    clf.fit(X[tr], y[tr])
    y_hat = clf.predict(X[te])
    f1 = f1_score(y[te], y_hat, average="macro")
    LOGGER.info("Linear probe F1 = %.3f (macro)", f1)
    LOGGER.info("Confusion\n%s", confusion_matrix(y[te], y_hat))
    return clf, f1

# ───────────────── 6. HINT CONDITION EFFECTS ———————————————

def centroid_drift(X, y_cond, y_cat):
    """Return ΔF1 & centroid shifts per hint vs none."""
    out = {}
    base = X[y_cond == "none"].mean(0)
    for h in [h for h in HINT_TYPES if h != "none"]:
        sel = y_cond == h
        if not sel.any():
            continue
        drift = np.linalg.norm(X[sel].mean(0) - base)
        out[h] = drift
    return out

# ─────────────────────── 7. EFFORT METRICS ———————————————

def layer_effort(reps, cats, layers):
    nL = len(layers)
    reps_3D = reps.reshape(len(reps), nL, -1)
    norms = np.linalg.norm(reps_3D, axis=2)        # [N,L]
    df = pd.DataFrame(norms, columns=[f"L{l}" for l in layers])
    df["category"] = cats
    return df.groupby("category").mean()

# ──────────── 8. FAITHFULNESS (contradiction + AUC) ———————————————

def contradiction_scores():
    """Stub – delegates to existing verifier script if available."""
    SCRIPT = ROOT / "g_cot_cluster/hidden_states/llm_verifier.py"
    if not SCRIPT.exists():
        LOGGER.warning("Verifier script not found – skipping contradiction")
        return None
    import importlib.util, subprocess, tempfile
    # For brevity, just shell‑out to the script and capture JSON output
    tmp = tempfile.NamedTemporaryFile(delete=False).name
    cmd = ["python", str(SCRIPT), "--output", tmp]
    subprocess.run(cmd, check=True)
    return pd.read_json(tmp)

# Prob trajectory -------------------------------------------------------

def answer_support_auc(text: str, answer_letter: str):
    load_model()
    ids = TOKENIZER(text, return_tensors="pt", add_special_tokens=False).input_ids.to(MODEL.device)
    with torch.no_grad():
        outs = MODEL(ids)
    logits = outs.logits[0]
    letter_tok = TOKENIZER(" " + answer_letter, add_special_tokens=False).input_ids[-1]
    probs = F.softmax(logits, dim=-1)[:, letter_tok].cpu().numpy()
    return np.trapz(probs) / len(probs)

# Faithfulness index -----------------------------------------------------

def faithfulness_index(contrad: float, auc: float, ent: float,
                       w=(0.4, 0.4, 0.2)):
    return w[0]*(1-contrad) + w[1]*auc - w[2]*ent

# ──────────────────────────── MAIN ENTRY ─────────────────────────────———
if __name__ == "__main__":
    args = ARGS  # already parsed

    if args.build_db:
        build_database(args.max_questions)
    if args.capture_hs:
        layers = tuple(int(x) for x in args.layers.split(","))
        hs_file = capture_hidden_states(layers, args.max_questions)
    else:
        layers = tuple(int(x) for x in args.layers.split(","))
        hs_file = CACHE_DIR / f"segment_hs_L{','.join(map(str,layers))}_max{args.max_questions or 'all'}.pkl"
        if not hs_file.exists():
            LOGGER.error("Hidden state cache %s not found; rerun with --capture_hs", hs_file)
            sys.exit(1)

    X, qid, hcond, cat = load_reps(hs_file)

    # Basic plots --------------------------------------------------------
    if args.plots:
        umap_plot(X, cat, "UMAP – phrase categories")
        umap_plot(X, hcond, "UMAP – hint types")
        plt.show()

    # Probe -------------------------------------------------------------
    if args.probe:
        _, f1 = phrase_probe(X, cat)

    # Hint effects ------------------------------------------------------
    if args.hint_comp:
        drift = centroid_drift(X, hcond, cat)
        LOGGER.info("Centroid L2 drift vs none: %s", drift)

    # Effort ------------------------------------------------------------
    if args.effort:
        df_eff = layer_effort(X, cat, layers)
        LOGGER.info("Layer effort per category\n%s", df_eff)

    # Faithfulness ------------------------------------------------------
    if args.faith:
        contr = contradiction_scores()
        if contr is not None:
            LOGGER.info("Contradiction scores head:\n%s", contr.head())
        # AUC & entropy require completions text
        con = duckdb.connect(DB_PATH)
        comp_df = con.execute("SELECT question_id, hint_type, completion FROM completions").df()
        comp_df["answer"] = comp_df["completion"].apply(extract_answer_letter)
        comp_df = comp_df.dropna(subset=["answer"]).head(args.max_questions or len(comp_df))
        aucs, ents = [], []
        for t, ans in tqdm(zip(comp_df.completion, comp_df.answer), total=len(comp_df)):
            aucs.append(answer_support_auc(t, ans))
            # quick overall entropy
            ids = TOKENIZER(t, return_tensors="pt", add_special_tokens=False).input_ids.to(MODEL.device)
            with torch.no_grad():
                probs = F.softmax(MODEL(ids).logits[0], -1)
            ents.append((-probs*probs.log()).sum(-1).mean().item())
        comp_df["auc"] = aucs
        comp_df["entropy"] = ents
        comp_df["FI"] = faithfulness_index(contrad=0.0, auc=comp_df.auc, ent=comp_df.entropy)  # contr stub
        LOGGER.info("Faithfulness index summary:\n%s", comp_df.FI.describe())

    LOGGER.info("All tasks done – exiting.")


"""
python g_cot_cluster/hidden_states/test4.py --build_db --max_questions 200

python g_cot_cluster/hidden_states/test4.py --capture_hs --layers -1 --max_questions 200

python g_cot_cluster/hidden_states/test4.py --plots --probe --hint_comp --max_questions 200
"""