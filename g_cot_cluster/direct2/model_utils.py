"""
model_utils.py
Simple explain‑then‑predict classifiers based on category sequences.
"""
from __future__ import annotations
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _seq_to_str(seq: List[str]) -> str:
    """Space‑separated category tokens → suitable for bag‑of‑n‑grams."""
    return " ".join(seq)


def prepare_xy(seq_df: pd.DataFrame, accuracy: pd.Series):
    """Construct **X, y** arrays for scikit‑learn.

    Parameters
    ----------
    seq_df : DataFrame
        Must contain columns `question_id`, `hint_type`, `category_sequence`.
    accuracy : Series
        Binary labels (0/1). Its index may be either:
        * **question_id**              – one label per question, or
        * **(question_id, hint_type)** – separate labels for each hint variant.

    Returns
    -------
    X : ndarray[str]
        Text representation of category sequences.
    y : ndarray[int]
        Accuracy labels.
    """

    # Decide join key dynamically ------------------------------------------
    if isinstance(accuracy.index, pd.MultiIndex):
        key_cols = list(accuracy.index.names)  # ['question_id', 'hint_type']
    else:
        key_cols = ["question_id"]

    # Merge sequences with labels (inner join keeps only labelled rows) -----
    merged = (
        seq_df.set_index(key_cols)[["category_sequence"]]
        .join(accuracy.rename("y"), how="inner")
        .dropna(subset=["y"])
        .reset_index()
    )

    X = merged["category_sequence"].apply(_seq_to_str).values
    y = merged["y"].astype(int).values
    return X, y

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xtp_logreg(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Train logistic regression on (uni+bi)gram bag‑of‑words of category
    sequences. Returns (pipeline, fpr, tpr, auc)."""
    if len(np.unique(y)) < 2:
        raise ValueError("Need both positive and negative examples for ROC/AUC.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipe = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train)
    y_score = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return pipe, fpr, tpr, auc(fpr, tpr)
