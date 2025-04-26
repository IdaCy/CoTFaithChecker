"""Statistical tests & Markov transition matrices."""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

################################################################################
# χ² & permutation tests on marginal category frequencies
################################################################################

def categorical_chi2(df: pd.DataFrame, category: str = "phrase_category") -> pd.DataFrame:
    """Return χ² test results for category frequency × hint_type."""
    # explode category sequences
    exploded = df[["hint_type", "category_sequence"]].explode("category_sequence")
    cont = pd.crosstab(exploded["hint_type"], exploded["category_sequence"])
    chi2, p, dof, exp = chi2_contingency(cont)
    return pd.DataFrame(
        {
            "chi2": [chi2],
            "dof": [dof],
            "p_value": [p],
        }
    )


def permutation_test(df: pd.DataFrame, n_runs: int = 10_000, seed: int | None = None) -> float:
    """Two‑sided permutation test on token counts between *hint* groups."""
    rng = np.random.default_rng(seed)
    grouped = [g["token_count"].to_numpy() for _, g in df.groupby("hint_type")]
    if len(grouped) != 2:
        raise ValueError("Permutation test currently supports exactly two groups.")
    x, y = grouped
    obs = abs(x.mean() - y.mean())
    concat = np.concatenate([x, y])
    n_x = len(x)
    more_extreme = 0
    for _ in range(n_runs):
        rng.shuffle(concat)
        diff = abs(concat[:n_x].mean() - concat[n_x:].mean())
        more_extreme += diff >= obs
    return more_extreme / n_runs

################################################################################
# Markov transition matrices
################################################################################

def _count_transitions(seq: list[str]) -> Counter[tuple[str, str]]:
    return Counter(zip(seq[:-1], seq[1:]))


def global_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    all_transitions: Counter[tuple[str, str]] = Counter()
    for seq in df["category_sequence"]:
        all_transitions.update(_count_transitions(seq))
    cats = sorted(set(itertools.chain.from_iterable(df["category_sequence"])) )
    mat = pd.DataFrame(0, index=cats, columns=cats, dtype=int)
    for (a, b), c in all_transitions.items():
        mat.loc[a, b] = c
    return mat


def plot_transition_heatmap(mat: pd.DataFrame, title: str = "Transition matrix") -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    plt.colorbar(im)
    plt.tight_layout()
