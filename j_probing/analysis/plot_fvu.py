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

def plot_combined_fvu(
    fvu_by_pos_layer: Dict[str, Dict[int, float]],
    model_name: str,
    n_layers: int,
    output_path: Path
):
    """Creates and saves the combined FVU plot using Seaborn styles."""
    # Apply Seaborn theme with improved aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(9, 8), dpi=300) # More square figure dimensions

    all_fvus = [
        fvu
        for fvu_by_layer in fvu_by_pos_layer.values()
        for fvu in fvu_by_layer.values()
        if not math.isnan(fvu) # Exclude NaNs for min/max calculation
    ]

    if not all_fvus:
        print("[Warning] No valid FVU data to plot.")
        plt.close()
        return

    max_fvu = max(all_fvus) if all_fvus else 1.0
    min_fvu = min(all_fvus) if all_fvus else 0.0
    
    # Better Y-axis limits - add only 5% padding above highest value
    plot_min_fvu = min(0.3, min_fvu * 0.9) if min_fvu < 1.0 else min_fvu * 0.95  # Start at 30% or 90% of min, whichever is smaller
    plot_max_fvu = max_fvu * 1.05  # Add just 5% padding above max

    # Custom Y-axis scale only if data spans across 1.0
    ax = plt.gca()
    if min_fvu < 1.0 and max_fvu > 1.0:
        transform = get_custom_scale_transform(plot_max_fvu, plot_min_fvu, z=0.7)
        ax.set_yscale(FuncScale(ax.yaxis, transform))
        
        # Define ticks more dynamically
        below_100_ticks = np.linspace(plot_min_fvu, 1.0, 8)  # 8 evenly spaced ticks below 100%
        
        # More sensible ticks above 100%
        if plot_max_fvu > 1.0:
            tick_step_above_1 = 0.1 if (plot_max_fvu - 1.0) <= 0.5 else 0.25  # Finer steps for small ranges
            above_100_ticks = np.arange(1.0, plot_max_fvu + tick_step_above_1, tick_step_above_1)
            yticks = np.unique(np.concatenate([below_100_ticks, above_100_ticks]))
        else:
            yticks = below_100_ticks
            
        # Limit number of ticks shown if too dense
        if len(yticks) > 15:
            yticks = yticks[::max(1, len(yticks)//15)]
        ax.set_yticks(yticks)
        
        # Explicitly set ylim to reduce whitespace
        ax.set_ylim(bottom=plot_min_fvu, top=plot_max_fvu)
    else:
        # Use linear scale with better limits
        ax.set_ylim(bottom=plot_min_fvu, top=plot_max_fvu)
        # Add more tick marks for better readability
        tick_step = (plot_max_fvu - plot_min_fvu) / 10
        yticks = np.arange(plot_min_fvu, plot_max_fvu + tick_step, tick_step)
        ax.set_yticks(yticks)

    # Define colors/styles with better contrast and visibility
    colors = {
        "assistant": "#45B8FE",  # Vibrant baby blue
        "think": "#FF9966",      # Salmon/coral orange (like in violin plot)
        "hint": "#1A7F64"        # Keep the same deep teal
    }
    
    # More distinctive line styles for better differentiation
    linestyles = {"assistant": "-", "think": "--", "hint": "-."}
    markers = {"assistant": "o", "think": "D", "hint": "s"}  # Changed think marker to diamond
    linewidths = {"assistant": 3.0, "think": 2.5, "hint": 2.5}
    markersize = {"assistant": 8, "think": 7, "hint": 7}  # Different sizes
    markeredgewidth = 1.5
    markeredgecolor = {"assistant": "white", "think": "white", "hint": "white"}

    # Plot each position
    layers_plotted = set()
    for pos_name in POSITION_ORDER:
        if pos_name in fvu_by_pos_layer:
            fvu_by_layer = fvu_by_pos_layer[pos_name]
            layers = sorted(fvu_by_layer.keys())
            # Filter out NaN fvus for plotting this specific line
            valid_layers = [l for l in layers if not math.isnan(fvu_by_layer[l])]
            valid_fvus = [fvu_by_layer[l] for l in valid_layers]
            
            if not valid_layers: # Skip plotting if no valid data for this position
                continue
                
            layers_plotted.update(valid_layers) # Keep track of all layers with data
            plt.plot(
                valid_layers,
                valid_fvus,
                label=pos_name,
                color=colors.get(pos_name, "black"),
                linestyle=linestyles.get(pos_name, "-"),
                linewidth=linewidths.get(pos_name, 1.8),
                marker=markers.get(pos_name, 'o'),
                markersize=markersize.get(pos_name, 7),  # Use the position-specific markersize
                markeredgewidth=markeredgewidth,
                markeredgecolor=markeredgecolor.get(pos_name, "white"),
                alpha=0.95 if pos_name == "assistant" else 0.9,  # Slightly different alpha for assistant
                zorder=3 if pos_name == "hint" else (2 if pos_name == "think" else 1)  # Control layer order
            )

    # Improved grid with better visibility
    ax.yaxis.grid(True, alpha=0.7, linestyle='-', color='#DDDDDD', linewidth=0.8)  # Stronger horizontal grid
    ax.xaxis.grid(True, alpha=0.3, linestyle='--', color='#DDDDDD', linewidth=0.5)  # Light vertical grid

    # Set x-ticks based on actual layers with data
    sorted_layers = sorted(list(layers_plotted))
    if sorted_layers:
        # Show reasonable number of tick labels
        step = 2 if n_layers <= 32 else 4
        tick_layers = list(range(0, n_layers, step))
        plt.xticks(tick_layers, rotation=0, fontsize=11)
        plt.xlim(left=min(sorted_layers)-0.5, right=max(sorted_layers)+0.5) # Add padding
    else:
        plt.xticks([]) # No data, no ticks

    # Format y-axis as percentages with larger font
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.yticks(fontsize=11)

    # Better label and title formatting with slightly smaller title to fit square plot
    plt.xlabel("Layer", fontsize=13, weight='bold')
    plt.ylabel("Fraction of Variance Unexplained (FVU)\n(lower is better)", fontsize=13, weight='bold')
    plt.title(f"{model_name} ({n_layers} layers)\nFVU by Layer and Token Position", fontsize=14, weight='bold', pad=10)
    
    # Position legend for minimal whitespace - upper right corner works better for square plot
    plt.legend(
        title="Token Position", 
        title_fontsize='12', 
        fontsize='11', 
        loc='upper left',  # Move to upper left
        framealpha=0.9,
        edgecolor='0.8',
        borderpad=0.8  # Add padding inside legend box
    )

    # Remove top/right spines but make remaining spines more visible
    sns.despine()
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('0.3')

    # Better layout adjustment
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


# --- Main Analysis Logic ---

def main():
    # --- Configuration --- 
    # TODO: Consider using argparse or a config file for flexibility
    DS_NAME = "mmlu"
    MODEL_NAME_FOR_PATH = "DeepSeek-R1-Distill-Llama-8B" # From train_probes
    HINT_TYPE = "sycophancy"
    N_QUESTIONS_STR = "5001" # From train_probes
    SPLIT_SEED = 42 # MUST match the seed used in train_probes.py
    VAL_FRAC = 0.10 # MUST match train_probes.py
    TEST_FRAC = 0.10 # MUST match train_probes.py

    # Construct paths
    BASE_DIR = Path("j_probing")
    DATA_DIR = BASE_DIR / "data" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    ACTS_DIR = BASE_DIR / "acts" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    PROBE_DIR = BASE_DIR / "probes" / MODEL_NAME_FOR_PATH / f"seed_{SPLIT_SEED}"
    # More descriptive filename
    plot_filename = f"{MODEL_NAME_FOR_PATH}_{DS_NAME}_{HINT_TYPE}_{N_QUESTIONS_STR}_seed{SPLIT_SEED}_fvu.png"
    OUTPUT_PLOT_PATH = BASE_DIR / "analysis" / "plots" / plot_filename

    # --- Load necessary data --- 
    target_map = load_target_data(DATA_DIR / "probing_data.json")
    all_qids_ordered, n_layers, d_model = load_question_ids(ACTS_DIR / "meta.json")

    # --- Determine test set labels and variance --- 
    _, _, test_qids = get_data_splits(
        all_qids_ordered, target_map, VAL_FRAC, TEST_FRAC, SPLIT_SEED
    )
    test_labels = np.array([target_map[qid] for qid in test_qids])
    variance_of_test_labels = np.var(test_labels)
    print(f"Variance of test set labels: {variance_of_test_labels:.4f}")

    if variance_of_test_labels < 1e-6: # Check for zero variance
        print("[Error] Variance of test labels is zero or near-zero. Cannot compute meaningful FVU.")
        return

    # --- Aggregate results and calculate FVU --- 
    fvu_results: Dict[str, Dict[int, float]] = {pos_name: {} for pos_name in POSITION_ORDER}

    for layer in range(n_layers):
        for pos_name in POSITION_ORDER:
            metrics_path = PROBE_DIR / f"layer_{layer:02d}" / pos_name / "metrics.json"
            if not metrics_path.exists():
                print(f"[Warning] Metrics file not found, skipping: {metrics_path}")
                continue
            
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
                test_loss_mse = metrics.get("test_loss")
                if test_loss_mse is not None:
                    fvu = test_loss_mse / variance_of_test_labels
                    fvu_results[pos_name][layer] = fvu
                else:
                    print(f"[Warning] 'test_loss' not found in {metrics_path}")
                    fvu_results[pos_name][layer] = float('nan') # Mark as NaN if missing

            except Exception as e:
                print(f"[Error] Failed to process metrics {metrics_path}: {e}")
                fvu_results[pos_name][layer] = float('nan') # Mark as NaN on error

    # --- Generate Plot --- 
    plot_combined_fvu(
        fvu_by_pos_layer=fvu_results,
        model_name=MODEL_NAME_FOR_PATH, # Or use a cleaner name if desired
        n_layers=n_layers,
        output_path=OUTPUT_PLOT_PATH
    )

if __name__ == "__main__":
    main() 