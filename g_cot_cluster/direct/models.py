"""Simple category‑sequence classifier → accuracy."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

################################################################################
# Helpers
################################################################################

def _seq_to_string(seq):
    """Convert category list to whitespace‑separated string for the vectorizer."""
    return " ".join(seq)

################################################################################
# Training / evaluation
################################################################################

def train_category_classifier(df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42) -> Tuple[LogisticRegression, CountVectorizer, pd.DataFrame]:
    """Train a logistic‑regression classifier to predict *is_correct* from category sequences."""
    df = df.dropna(subset=["is_correct"])
    X_str = df["category_sequence"].map(_seq_to_string)
    y = df["is_correct"].astype(int)

    vec = CountVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(X_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1_000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results = pd.DataFrame(
        {
            "set": ["test"],
            "accuracy": [acc],
        }
    )

    # ---- plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("Confusion matrix – category‑sequence classifier")
    plt.colorbar(im)
    plt.tight_layout()

    return clf, vec, results


def evaluate_classifier(clf: LogisticRegression, vec: CountVectorizer, df: pd.DataFrame) -> float:
    df = df.dropna(subset=["is_correct"])
    X = vec.transform(df["category_sequence"].map(_seq_to_string))
    y = df["is_correct"].astype(int)
    return accuracy_score(y, clf.predict(X))