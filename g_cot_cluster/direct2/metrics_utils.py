"""
metrics_utils.py
Functions to compute descriptive and inferential metrics for CoT faithfulness
experiments. Includes robustness for environments where the NLTK *punkt* model
is not pre‑installed.
"""
from __future__ import annotations
import itertools, math
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, pointbiserialr
import nltk

# ---------------------------------------------------------------------------
# Safe tokenisation helpers (gracefully handle missing punkt data)
# ---------------------------------------------------------------------------

from nltk.tokenize import word_tokenize


def _safe_word_tokenize(text: str):
    """Robust tokeniser.

    1. Try NLTK's *word_tokenize* (needs the *punkt* resource).
    2. If *punkt* is missing *and* cannot be downloaded (e.g. offline
       environment), fall back to a simple ``text.split()`` whitespace split –
       good enough for length / lexical‑diversity statistics.
    """
    try:
        return word_tokenize(text)
    except LookupError:
        # Attempt a silent, best‑effort download *once*.
        try:
            nltk.download("punkt", quiet=True, raise_on_error=False)
            return word_tokenize(text)
        except LookupError:
            # Still unavailable (e.g. no internet) – degrade gracefully.
            return text.split()# ---------------------------------------------------------------------------
# Category frequency & sequence patterns
# ---------------------------------------------------------------------------

def category_frequencies(seq_df: pd.DataFrame) -> pd.DataFrame:
    """Return a *long* DataFrame containing counts for every
    ``phrase_category`` / ``hint_type`` combination."""
    records = []
    for hint, group in seq_df.groupby("hint_type"):
        counts = Counter(itertools.chain.from_iterable(group["category_sequence"]))
        records.extend(
            {"hint_type": hint, "phrase_category": cat, "count": n}
            for cat, n in counts.items()
        )
    return pd.DataFrame(records)


def markov_transition_matrix(
    seq_df: pd.DataFrame, categories: List[str]
) -> Dict[str, pd.DataFrame]:
    """Compute (row‑normalised) first‑order transition matrices for each
    ``hint_type``."""
    matrices: Dict[str, pd.DataFrame] = {}
    for hint, group in seq_df.groupby("hint_type"):
        cnt = Counter()
        for seq in group["category_sequence"]:
            cnt.update(zip(seq, seq[1:]))
        mat = pd.DataFrame(0.0, index=categories, columns=categories)
        for (a, b), n in cnt.items():
            mat.loc[a, b] = n
        row_sums = mat.sum(axis=1).replace(0, np.nan)
        mat = mat.div(row_sums, axis=0).fillna(0.0)
        matrices[hint] = mat
    return matrices


def bigram_distributions(seq_df: pd.DataFrame) -> Dict[str, Counter]:
    """Return un‑normalised bigram *Counter* for each ``hint_type``."""
    d: Dict[str, Counter] = {}
    for hint, group in seq_df.groupby("hint_type"):
        c = Counter()
        for seq in group["category_sequence"]:
            c.update(zip(seq, seq[1:]))
        d[hint] = c
    return d


def js_divergence_matrix(
    bigram_counts: Dict[str, Counter], categories: List[str]
) -> pd.DataFrame:
    """Pairwise Jensen–Shannon divergence of bigram distributions (base‑2)."""
    hints = list(bigram_counts)
    all_bigrams = [(a, b) for a in categories for b in categories]
    vecs = {}
    for hint, c in bigram_counts.items():
        total = sum(c.values())
        prob = np.array([c.get(bg, 0) / total if total else 0 for bg in all_bigrams])
        vecs[hint] = prob
    js = np.zeros((len(hints), len(hints)))
    for i, h1 in enumerate(hints):
        for j, h2 in enumerate(hints):
            if i <= j:
                d = jensenshannon(vecs[h1], vecs[h2], base=2)
                js[i, j] = js[j, i] = d
    return pd.DataFrame(js, index=hints, columns=hints)

# ---------------------------------------------------------------------------
# Length / entropy metrics
# ---------------------------------------------------------------------------

def _yules_k(tokens: List[str]) -> float:
    """Yule's K lexical diversity."""
    freq = Counter(tokens)
    M1 = len(tokens)
    M2 = sum(f * f for f in freq.values())
    if M1 == 0:
        return math.nan
    return 10000 * (M2 - M1) / (M1 * M1)


def length_entropy_metrics(seq_df: pd.DataFrame) -> pd.DataFrame:
    """Compute token_count, category_entropy, Yule's K and presence of
    *backtracking_revision* for every *question_id* / *hint_type* pair."""
    rows: List[dict] = []
    for row in seq_df.itertuples():
        tokens = _safe_word_tokenize(row.full_text)
        cat_counts = Counter(row.category_sequence)
        probs = np.fromiter(cat_counts.values(), float) / len(row.category_sequence)
        rows.append(
            {
                "question_id": row.question_id,
                "hint_type": row.hint_type,
                "token_count": len(tokens),
                "category_entropy": entropy(probs, base=2),
                "yules_k": _yules_k(tokens),
                "backtracking": int("backtracking_revision" in row.category_sequence),
            }
        )
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Backtracking & accuracy
# ---------------------------------------------------------------------------

def backtracking_accuracy_correlation(
    metrics_df: pd.DataFrame, accuracy: pd.Series
):
    """Return point‑biserial *r* and *p* between ``backtracking`` (binary) and
    ``accuracy`` (binary). *accuracy* must be indexed by ``question_id``."""
    merged = metrics_df.set_index("question_id").join(accuracy.rename("accuracy"))
    merged = merged.dropna(subset=["accuracy"])
    r, p = pointbiserialr(merged["backtracking"].astype(int), merged["accuracy"].astype(int))
    return r, p
