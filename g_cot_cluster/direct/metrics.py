"""Chain‑of‑thought metrics: token counts, entropy, lexical diversity."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# Basic numeric metrics
################################################################################

def _tokenize_whitespace(text: str) -> list[str]:
    return text.split()


def token_count(text: str) -> int:
    return len(_tokenize_whitespace(text))


def sequence_entropy(seq: Sequence[str]) -> float:
    cnt = Counter(seq)
    n = sum(cnt.values())
    return -sum((c / n) * math.log2(c / n) for c in cnt.values() if c)


def yules_k(text: str) -> float:
    words = _tokenize_whitespace(text.lower())
    if not words:
        return 0.0
    freq = Counter(words)
    m1 = len(words)
    m2 = sum(f * f for f in freq.values())
    return (10_000 * (m2 - m1)) / (m1 * m1)


################################################################################
# Data‑frame helpers & charting
################################################################################

def add_basic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add token_count, category_entropy, yules_k columns."""
    df = df.copy()
    df["token_count"] = df["full_text"].map(token_count)
    df["category_entropy"] = df["category_sequence"].map(sequence_entropy)
    df["yules_k"] = df["full_text"].map(yules_k)
    return df


def _boxplot(df: pd.DataFrame, column: str, title: str) -> None:
    plt.figure()
    (df[["hint_type", column]].boxplot(by="hint_type"))
    plt.suptitle("")
    plt.title(title)
    plt.xlabel("Hint type")
    plt.ylabel(column.replace("_", " ").title())
    plt.tight_layout()


def plot_token_counts(df: pd.DataFrame) -> None:
    _boxplot(df, "token_count", "Token count distribution by hint type")


def plot_category_entropy(df: pd.DataFrame) -> None:
    _boxplot(df, "category_entropy", "Category entropy by hint type")


def plot_lexical_diversity(df: pd.DataFrame) -> None:
    _boxplot(df, "yules_k", "Yule's K lexical diversity by hint type")