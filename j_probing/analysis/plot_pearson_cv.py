#!/usr/bin/env python3
import json
import random
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
import torch # Although we don't train, some utils might need it
import matplotlib.pyplot as plt
from matplotlib.scale import FuncScale
from matplotlib.ticker import FuncFormatter
from numpy.typing import ArrayLike
import seaborn as sns # Import Seaborn

# --- Assuming utils are importable or defined here ---
# Simplified imports for clarity, assuming these exist in utils
# from ..utils.training_utils import setup_determinism, load_target_data, load_question_ids

# --- Utility functions copied for simplicity (remove if importing) ---
def setup_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_target_data(path: Path) -> Dict[int, float]:
    target_map = {}
    print(f"Loading target data from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Target data file not found: {path}")
        raise
    for record in data:
        qid = record["question_id"]
        prob = record.get("prob_verb_match")
        if prob is not None: target_map[qid] = float(prob)
    if not target_map: print(f"[Warning] No valid 'prob_verb_match' values loaded from {path}")
    return target_map

def load_question_ids(meta_path: Path) -> List[int]:
    print(f"Loading question ID order from: {meta_path}")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Meta file not found: {meta_path}"); raise
    qids = meta_data.get("question_ids")
    n_layers = meta_data.get("n_layers")
    d_model = meta_data.get("d_model")
    if qids is None or n_layers is None or d_model is None: raise ValueError(f"Meta missing fields: {meta_path}")
    if not isinstance(qids, list): raise ValueError(f"'question_ids' not a list: {meta_path}")
    print(f"Found {len(qids)} QIDs, {n_layers} layers, d_model {d_model}")
    return qids, n_layers, d_model
# --- End Copied Utils ---

POSITION_ORDER = ["assistant", "think", "hint"]
POSITION_MAP = {0: "assistant", 1: "think", 2: "hint"}

# Removed get_custom_scale_transform as it's likely not needed for Pearson r

def plot_combined_pearson_cv(
    pearson_mean_by_pos_layer: Dict[str, Dict[int, float]],
    pearson_std_by_pos_layer: Dict[str, Dict[int, float]],
    model_name: str,
    n_layers: int,
    k_folds: int,
    cv_seed: int,
    output_path: Path,
):
    """Creates and saves the combined CV Pearson Correlation plot with error bands."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(9, 8), dpi=300)

    all_pearson_means = [
        r
        for r_means in pearson_mean_by_pos_layer.values()
        for r in r_means.values()
        if not math.isnan(r)
    ]
    all_pearson_uppers = [
        r_mean + r_std
        for pos, r_means in pearson_mean_by_pos_layer.items()
        for layer, r_mean in r_means.items()
        if not math.isnan(r_mean) and not math.isnan(r_std := pearson_std_by_pos_layer.get(pos, {}).get(layer, float('nan')))
    ]
    all_pearson_lowers = [
        r_mean - r_std
        for pos, r_means in pearson_mean_by_pos_layer.items()
        for layer, r_mean in r_means.items()
        if not math.isnan(r_mean) and not math.isnan(r_std := pearson_std_by_pos_layer.get(pos, {}).get(layer, float('nan')))
    ]

    if not all_pearson_means:
        print("[Warning] No valid Pearson Correlation data to plot."); plt.close(); return

    # Determine plot bounds based on mean +/- std, constrained within [-1, 1]
    max_r_upper = min(1.0, max(all_pearson_uppers)) if all_pearson_uppers else 1.0
    min_r_lower = max(-1.0, min(all_pearson_lowers)) if all_pearson_lowers else -1.0

    # Sensible Y-axis limits, likely 0 to 1 for this task, but allow slight padding
    plot_min_r = max(-0.05, min_r_lower - 0.05) # Start slightly below 0 or lowest value
    plot_max_r = min(1.05, max_r_upper + 0.05) # Go slightly above 1 or highest value

    # --- Y-axis Setup (Linear Scale) ---
    ax = plt.gca()
    ax.set_ylim(bottom=plot_min_r, top=plot_max_r)
    # Define ticks for Pearson r (e.g., every 0.1 or 0.2)
    tick_step = 0.1 if (plot_max_r - plot_min_r) <= 1.2 else 0.2
    # Ensure ticks don't go beyond [-1, 1]
    yticks = np.arange(max(-1.0, math.ceil(plot_min_r * 10) / 10), # Start at nearest 0.1 above min
                       min(1.01, plot_max_r), # Stop at or just above max (up to 1.0)
                       tick_step)
    ax.set_yticks(yticks)
    # Format Y axis ticks (e.g., 0.5, 0.6, etc.)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    # --- End Y-axis Setup ---

    # Define colors/styles (same as before)
    colors = {"assistant": "#45B8FE", "think": "#FF9966", "hint": "#1A7F64"}
    linestyles = {"assistant": "-", "think": "--", "hint": "-."}
    markers = {"assistant": "o", "think": "D", "hint": "s"}
    linewidths = {"assistant": 3.0, "think": 2.5, "hint": 2.5}
    markersize = {"assistant": 8, "think": 7, "hint": 7}
    markeredgewidth = 1.5
    markeredgecolor = {"assistant": "white", "think": "white", "hint": "white"}

    layers_plotted = set()
    for pos_name in POSITION_ORDER:
        if pos_name in pearson_mean_by_pos_layer:
            r_means = pearson_mean_by_pos_layer[pos_name]
            r_stds = pearson_std_by_pos_layer.get(pos_name, {})

            layers = sorted(r_means.keys())
            valid_layers = [l for l in layers if not math.isnan(r_means[l])]
            valid_r_means = np.array([r_means[l] for l in valid_layers])
            valid_r_stds = np.array([r_stds.get(l, 0.0) for l in valid_layers]) # Default std to 0

            if not valid_layers: continue
            layers_plotted.update(valid_layers)

            # Plot the mean line
            plt.plot(
                valid_layers,
                valid_r_means,
                label=pos_name,
                color=colors.get(pos_name, "black"),
                linestyle=linestyles.get(pos_name, "-"),
                linewidth=linewidths.get(pos_name, 1.8),
                marker=markers.get(pos_name, 'o'),
                markersize=markersize.get(pos_name, 7),
                markeredgewidth=markeredgewidth,
                markeredgecolor=markeredgecolor.get(pos_name, "white"),
                alpha=0.95,
                zorder=3 if pos_name == "hint" else (2 if pos_name == "think" else 1)
            )

            # Plot the shaded error band (mean +/- std dev)
            plt.fill_between(
                valid_layers,
                valid_r_means - valid_r_stds,
                valid_r_means + valid_r_stds,
                color=colors.get(pos_name, "black"),
                alpha=0.15, # Lighter shade for the band
                zorder=0 # Draw bands behind lines
            )

    # Grid, Ticks, Labels, Title, Legend (adjust title/labels for Pearson)
    ax.yaxis.grid(True, alpha=0.7, linestyle='-', color='#DDDDDD', linewidth=0.8)
    ax.xaxis.grid(True, alpha=0.3, linestyle='--', color='#DDDDDD', linewidth=0.5)

    sorted_layers = sorted(list(layers_plotted))
    if sorted_layers:
        step = 2 if n_layers <= 32 else 4
        tick_layers = list(range(0, n_layers, step))
        plt.xticks(tick_layers, rotation=0, fontsize=11)
        plt.xlim(left=min(sorted_layers)-0.5, right=max(sorted_layers)+0.5)
    else: plt.xticks([])

    plt.yticks(fontsize=11)

    plt.xlabel("Layer", fontsize=13, weight='bold')
    plt.ylabel("Pearson Correlation (r)", fontsize=13, weight='bold') # Updated Y label
    plt.title(f"{k_folds}-Fold CV Pearson Correlation by Layer and Token Position ({n_layers} layers)\nCV Seed {cv_seed}", fontsize=14, weight='bold', pad=10) # Updated title
    
    plt.legend(title="Token Position", title_fontsize='12', fontsize='11',
               loc='best', framealpha=0.9, edgecolor='0.8', borderpad=0.8) # Use 'best' location

    sns.despine()
    for spine in ['left', 'bottom']: ax.spines[spine].set_linewidth(1.2); ax.spines[spine].set_color('0.3')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


# --- Main CV Analysis Logic ---

def main():
    # --- Configuration (Should match train_probes_cv.py settings) ---
    DS_NAME = "mmlu"
    MODEL_NAME_FOR_PATH = "DeepSeek-R1-Distill-Llama-8B"
    HINT_TYPE = "sycophancy"
    N_QUESTIONS_STR = "5001"
    K_FOLDS = 5 # Number of folds used during training
    CV_SPLIT_SEED = 42 # Seed used for the KFold split

    BASE_DIR = Path("j_probing")
    # Path to original data needed meta info
    DATA_DIR = BASE_DIR / "data" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    ACTS_DIR = BASE_DIR / "acts" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    # Path to the CV results directory
    PROBE_DIR = BASE_DIR / "probes" / MODEL_NAME_FOR_PATH / f"cv_k{K_FOLDS}" / f"seed_{CV_SPLIT_SEED}"

    plot_filename = f"{MODEL_NAME_FOR_PATH}_{DS_NAME}_{HINT_TYPE}_{N_QUESTIONS_STR}_k{K_FOLDS}_seed{CV_SPLIT_SEED}_pearson_cv.png" # Updated filename
    OUTPUT_PLOT_PATH = BASE_DIR / "analysis" / "plots" / plot_filename

    # --- Load necessary base data --- 
    # target_map = load_target_data(DATA_DIR / "probing_data.json") # Don't need targets for Pearson plot
    all_qids_ordered, n_layers, d_model = load_question_ids(ACTS_DIR / "meta.json")

    # --- Load and process CV results --- 
    pearson_mean_results: Dict[str, Dict[int, float]] = {pos: {} for pos in POSITION_ORDER}
    pearson_std_results: Dict[str, Dict[int, float]] = {pos: {} for pos in POSITION_ORDER}

    results_summary_path = PROBE_DIR / "all_cv_results_summary.json"
    if not results_summary_path.exists():
        print(f"[Error] CV Results summary file not found: {results_summary_path}"); return

    with open(results_summary_path, "r") as f:
        all_cv_results = json.load(f)

    for layer_pos_key, metrics in all_cv_results.items():
        # Extract layer and position name
        try:
            parts = layer_pos_key.split("_")
            layer = int(parts[0][1:]) # Assumes L{layer}_...
            pos_name = "_".join(parts[1:])
            if pos_name not in POSITION_ORDER:
                 print(f"[Warning] Skipping unknown position in key: {layer_pos_key}"); continue
        except (IndexError, ValueError):
            print(f"[Warning] Could not parse layer/pos from key: {layer_pos_key}"); continue

        # Extract Pearson metrics
        mean_pearson_r = metrics.get("test_pearson_r_mean")
        std_pearson_r = metrics.get("test_pearson_r_std", float('nan'))

        if mean_pearson_r is not None and not math.isnan(mean_pearson_r):
            pearson_mean_results[pos_name][layer] = mean_pearson_r
            pearson_std_results[pos_name][layer] = std_pearson_r if not math.isnan(std_pearson_r) else 0.0 # Default std to 0 if NaN
        else:
            print(f"[Warning] Mean Pearson R missing or NaN for {layer_pos_key}")
            pearson_mean_results[pos_name][layer] = float('nan')
            pearson_std_results[pos_name][layer] = float('nan')

    # --- Generate Plot --- 
    plot_combined_pearson_cv(
        pearson_mean_by_pos_layer=pearson_mean_results,
        pearson_std_by_pos_layer=pearson_std_results,
        model_name=MODEL_NAME_FOR_PATH,
        n_layers=n_layers,
        k_folds=K_FOLDS,
        cv_seed=CV_SPLIT_SEED,
        output_path=OUTPUT_PLOT_PATH,
    )

if __name__ == "__main__":
    main() 