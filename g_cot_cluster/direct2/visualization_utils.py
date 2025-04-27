"""
visualization_utils.py
High‑level plotting helpers for the CoT faithfulness notebook.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Category frequency & transitions
# ---------------------------------------------------------------------------

def bar_category_freq(freq_df: pd.DataFrame, normalize: bool = True):
    data = freq_df.copy()
    if normalize:
        totals = data.groupby("hint_type")["count"].transform("sum")
        data["freq"] = data["count"] / totals
    else:
        data["freq"] = data["count"]
    g = sns.catplot(
        data=data,
        x="phrase_category",
        y="freq",
        hue="hint_type",
        kind="bar",
        height=5,
        aspect=2,
    )
    g.set_xticklabels(rotation=45, ha="right")
    g.set_axis_labels("Phrase category", "Relative frequency" if normalize else "Count")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Category distribution by hint type")
    return g


def heatmap_transition(matrix: pd.DataFrame, title: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap="mako", square=True, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("→ To category")
    plt.ylabel("From category →")
    plt.tight_layout()


def heatmap_js(js_mat: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    sns.heatmap(js_mat, annot=True, cmap="crest", vmin=0, vmax=1)
    plt.title("JS divergence between bigram distributions")
    plt.tight_layout()

# ---------------------------------------------------------------------------
# Length & backtracking visuals
# ---------------------------------------------------------------------------

def dist_length(metrics_df: pd.DataFrame):
    g = sns.displot(
        data=metrics_df,
        x="token_count",
        hue="hint_type",
        kind="kde",
        height=5,
        aspect=1.6,
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Token‑count distribution by hint type")
    return g


def scatter_backtracking(metrics_df: pd.DataFrame, accuracy):
    merged = (
        metrics_df.set_index("question_id")
        .join(accuracy.rename("accuracy"))
        .dropna(subset=["accuracy"])
    )
    plt.figure(figsize=(4, 4))
    sns.stripplot(
        data=merged,
        x="backtracking",
        y="accuracy",
        jitter=0.25,
        alpha=0.7,
    )
    plt.xticks([0, 1], ["No backtracking", "Backtracking"])
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.title("Accuracy conditioned on backtracking")
    plt.tight_layout()

# ---------------------------------------------------------------------------
# Classifier ROC
# ---------------------------------------------------------------------------

def plot_roc(fpr, tpr, label: str):
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Explain‑then‑predict ROC")
    plt.legend()
    plt.tight_layout()
