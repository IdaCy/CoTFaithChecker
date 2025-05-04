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

# --- Copy relevant utility functions or import if refactored ---
# Assuming these functions are either here or accessible via import
# (Copied from train_probes.py for now)
def setup_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_target_data(path: Path) -> Dict[int, float]:
    """Loads probing_data.json and returns a map from question_id to prob_verb_match."""
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
        if prob is not None: # Filter out null probabilities
             target_map[qid] = float(prob)
    if not target_map:
        print(f"[Warning] No valid 'prob_verb_match' values loaded from {path}")
    return target_map

def load_question_ids(meta_path: Path) -> List[int]:
    """Loads the meta.json file and returns the ordered list of question IDs."""
    print(f"Loading question ID order from: {meta_path}")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Meta file not found: {meta_path}")
        raise
    qids = meta_data.get("question_ids")
    n_layers = meta_data.get("n_layers") # Load n_layers as well
    d_model = meta_data.get("d_model") # Load d_model as well
    if qids is None or n_layers is None or d_model is None:
        raise ValueError(f"'question_ids', 'n_layers' or 'd_model' not found in {meta_path}")
    if not isinstance(qids, list):
         raise ValueError(f"'question_ids' in {meta_path} is not a list")
    print(f"Found {len(qids)} QIDs, {n_layers} layers, d_model {d_model}")
    return qids, n_layers, d_model # Return dimensions too

def get_data_splits(
    all_qids: List[int],
    target_map: Dict[int, float],
    val_frac: float,
    test_frac: float,
    seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """Splits question IDs into train, validation, and test sets randomly."""
    valid_qids = [qid for qid in all_qids if qid in target_map]
    n_total = len(valid_qids)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Dataset size ({n_total}) too small for specified splits.")
    setup_determinism(seed)
    shuffled_qids = random.sample(valid_qids, n_total)
    test_qids = shuffled_qids[:n_test]
    val_qids = shuffled_qids[n_test : n_test + n_val]
    train_qids = shuffled_qids[n_test + n_val :]
    return train_qids, val_qids, test_qids

# --- Plotting Configuration & Helpers (Adapted) ---

POSITION_ORDER = ["assistant", "think", "hint"]
POSITION_MAP = {0: "assistant", 1: "think", 2: "hint"}

def get_custom_scale_transform(
    max_value: float,
    min_value: float,
    z: float = 0.7, # Proportion of axis for range [min_val, 1]
) -> tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]:
    """Create custom scale transformation for y-axis.
    Compresses the range > 1.0.
    Args:
        max_value: The maximum value expected on the axis.
        min_value: The minimum value expected on the axis (must be < 1.0).
        z: The proportion of the plot height dedicated to the [min_value, 1.0] range.
    """
    if min_value >= 1.0:
         # If all data is >= 100%, use linear scale
         def identity(y): return np.asarray(y)
         return identity, identity

    def forward(y: ArrayLike) -> ArrayLike:
        y_arr = np.asarray(y)
        scaled = np.where(
            y_arr <= 1,
            z * (y_arr - min_value) / (1 - min_value), # Scale [min_value, 1] to [0, z]
            z + (1 - z) * (y_arr - 1) / (max_value - 1) if max_value > 1 else z # Scale [1, max_value] to [z, 1]
        )
        return scaled

    def inverse(y: ArrayLike) -> ArrayLike:
        y_arr = np.asarray(y)
        inversed = np.where(
            y_arr <= z,
            min_value + y_arr * (1 - min_value) / z, # Inverse map [0, z] to [min_value, 1]
            1 + (y_arr - z) * (max_value - 1) / (1 - z) if max_value > 1 else 1 # Inverse map [z, 1] to [1, max_value]
        )
        return inversed

    return forward, inverse

def plot_combined_fvu_cv(
    fvu_mean_by_pos_layer: Dict[str, Dict[int, float]],
    fvu_std_by_pos_layer: Dict[str, Dict[int, float]],
    model_name: str,
    n_layers: int,
    k_folds: int,
    cv_seed: int,
    output_path: Path,
):
    """Creates and saves the combined CV FVU plot with error bands."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(9, 8), dpi=300)

    all_fvu_means = [
        fvu
        for fvu_means in fvu_mean_by_pos_layer.values()
        for fvu in fvu_means.values()
        if not math.isnan(fvu)
    ]
    all_fvu_uppers = [
        fvu_mean + fvu_std
        for pos, fvu_means in fvu_mean_by_pos_layer.items()
        for layer, fvu_mean in fvu_means.items()
        if not math.isnan(fvu_mean) and not math.isnan(fvu_std := fvu_std_by_pos_layer.get(pos, {}).get(layer, float('nan')))
    ]
    all_fvu_lowers = [
        fvu_mean - fvu_std
        for pos, fvu_means in fvu_mean_by_pos_layer.items()
        for layer, fvu_mean in fvu_means.items()
        if not math.isnan(fvu_mean) and not math.isnan(fvu_std := fvu_std_by_pos_layer.get(pos, {}).get(layer, float('nan')))
    ]

    if not all_fvu_means:
        print("[Warning] No valid FVU data to plot."); plt.close(); return

    # Determine plot bounds based on mean +/- std
    max_fvu_upper = max(all_fvu_uppers) if all_fvu_uppers else 1.0
    min_fvu_lower = min(all_fvu_lowers) if all_fvu_lowers else 0.0

    plot_min_fvu = min(0.3, min_fvu_lower * 0.9) if min_fvu_lower < 1.0 else min_fvu_lower * 0.95
    plot_max_fvu = max_fvu_upper * 1.05

    # --- Y-axis Scaling (same as before, using plot_min/max derived from bounds) ---
    ax = plt.gca()
    min_mean_fvu = min(all_fvu_means)
    max_mean_fvu = max(all_fvu_means)
    if min_mean_fvu < 1.0 and max_mean_fvu > 1.0:
        transform = get_custom_scale_transform(plot_max_fvu, plot_min_fvu, z=0.7)
        ax.set_yscale(FuncScale(ax.yaxis, transform))
        below_100_ticks = np.linspace(plot_min_fvu, 1.0, 8)
        if plot_max_fvu > 1.0:
            tick_step_above_1 = 0.1 if (plot_max_fvu - 1.0) <= 0.5 else 0.25
            above_100_ticks = np.arange(1.0, plot_max_fvu + tick_step_above_1, tick_step_above_1)
            yticks = np.unique(np.concatenate([below_100_ticks, above_100_ticks]))
        else: yticks = below_100_ticks
        if len(yticks) > 15: yticks = yticks[::max(1, len(yticks)//15)]
        ax.set_yticks(yticks)
        ax.set_ylim(bottom=plot_min_fvu, top=plot_max_fvu)
    else:
        ax.set_ylim(bottom=plot_min_fvu, top=plot_max_fvu)
        tick_step = (plot_max_fvu - plot_min_fvu) / 10
        yticks = np.arange(plot_min_fvu, plot_max_fvu + tick_step, tick_step)
        ax.set_yticks(yticks)
    # --- End Y-axis Scaling ---

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
        if pos_name in fvu_mean_by_pos_layer:
            fvu_means = fvu_mean_by_pos_layer[pos_name]
            fvu_stds = fvu_std_by_pos_layer.get(pos_name, {})

            layers = sorted(fvu_means.keys())
            valid_layers = [l for l in layers if not math.isnan(fvu_means[l])]
            valid_fvu_means = np.array([fvu_means[l] for l in valid_layers])
            valid_fvu_stds = np.array([fvu_stds.get(l, 0.0) for l in valid_layers]) # Default std to 0 if missing

            if not valid_layers: continue
            layers_plotted.update(valid_layers)

            # Plot the mean line
            plt.plot(
                valid_layers,
                valid_fvu_means,
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
                valid_fvu_means - valid_fvu_stds,
                valid_fvu_means + valid_fvu_stds,
                color=colors.get(pos_name, "black"),
                alpha=0.15, # Lighter shade for the band
                zorder=0 # Draw bands behind lines
            )

    # Grid, Ticks, Labels, Title, Legend (adjust title for CV)
    ax.yaxis.grid(True, alpha=0.7, linestyle='-', color='#DDDDDD', linewidth=0.8)
    ax.xaxis.grid(True, alpha=0.3, linestyle='--', color='#DDDDDD', linewidth=0.5)

    sorted_layers = sorted(list(layers_plotted))
    if sorted_layers:
        step = 2 if n_layers <= 32 else 4
        tick_layers = list(range(0, n_layers, step))
        plt.xticks(tick_layers, rotation=0, fontsize=11)
        plt.xlim(left=min(sorted_layers)-0.5, right=max(sorted_layers)+0.5)
    else: plt.xticks([])

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.yticks(fontsize=11)

    plt.xlabel("Layer", fontsize=13, weight='bold')
    plt.ylabel("Fraction of Variance Unexplained (FVU)\n(lower is better)", fontsize=13, weight='bold')
    plt.title(f"{k_folds}-Fold CV FVU by Layer and Token Position ({n_layers} layers)\nCV Seed {cv_seed}", fontsize=14, weight='bold', pad=10)
    
    plt.legend(title="Token Position", title_fontsize='12', fontsize='11',
               loc='upper left', framealpha=0.9, edgecolor='0.8', borderpad=0.8)

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
    # Path to original data needed for label variance and meta info
    DATA_DIR = BASE_DIR / "data" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    ACTS_DIR = BASE_DIR / "acts" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    # Path to the CV results directory
    PROBE_DIR = BASE_DIR / "probes" / MODEL_NAME_FOR_PATH / f"cv_k{K_FOLDS}" / f"seed_{CV_SPLIT_SEED}"

    plot_filename = f"{MODEL_NAME_FOR_PATH}_{DS_NAME}_{HINT_TYPE}_{N_QUESTIONS_STR}_k{K_FOLDS}_seed{CV_SPLIT_SEED}_fvu_cv.png"
    OUTPUT_PLOT_PATH = BASE_DIR / "analysis" / "plots" / plot_filename

    # --- Load necessary base data --- 
    target_map = load_target_data(DATA_DIR / "probing_data.json")
    all_qids_ordered, n_layers, d_model = load_question_ids(ACTS_DIR / "meta.json")

    # --- Calculate variance across ALL valid labels --- 
    all_valid_labels = np.array([prob for qid, prob in target_map.items() if qid in all_qids_ordered and prob is not None])
    variance_of_all_labels = np.var(all_valid_labels)
    print(f"Variance of all labels: {variance_of_all_labels:.4f}")

    if variance_of_all_labels < 1e-6: # Check for zero variance
        print("[Error] Variance of labels is zero. Cannot compute FVU."); return

    # --- Load and process CV results --- 
    fvu_mean_results: Dict[str, Dict[int, float]] = {pos: {} for pos in POSITION_ORDER}
    fvu_std_results: Dict[str, Dict[int, float]] = {pos: {} for pos in POSITION_ORDER}

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
            pos_name = "_".join(parts[1:]) # Handle potential underscores in pos_name if any
            if pos_name not in POSITION_ORDER:
                 print(f"[Warning] Skipping unknown position in key: {layer_pos_key}")
                 continue
        except (IndexError, ValueError):
            print(f"[Warning] Could not parse layer/pos from key: {layer_pos_key}")
            continue

        mean_test_loss = metrics.get("test_loss_mean")
        std_test_loss = metrics.get("test_loss_std", float('nan')) # Default std to NaN if missing

        if mean_test_loss is not None and not math.isnan(mean_test_loss):
            fvu_mean = mean_test_loss / variance_of_all_labels
            # Propagate error: std(FVU) = std(Loss) / Var(Labels)
            fvu_std = std_test_loss / variance_of_all_labels if not math.isnan(std_test_loss) else float('nan')
            
            fvu_mean_results[pos_name][layer] = fvu_mean
            fvu_std_results[pos_name][layer] = fvu_std
        else:
            print(f"[Warning] Mean test loss missing or NaN for {layer_pos_key}")
            fvu_mean_results[pos_name][layer] = float('nan')
            fvu_std_results[pos_name][layer] = float('nan')

    # --- Generate Plot --- 
    plot_combined_fvu_cv(
        fvu_mean_by_pos_layer=fvu_mean_results,
        fvu_std_by_pos_layer=fvu_std_results,
        model_name=MODEL_NAME_FOR_PATH,
        n_layers=n_layers,
        k_folds=K_FOLDS,
        cv_seed=CV_SPLIT_SEED,
        output_path=OUTPUT_PLOT_PATH,
    )

if __name__ == "__main__":
    main() 