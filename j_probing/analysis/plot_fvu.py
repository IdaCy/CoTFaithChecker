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
    # ---> MODIFICATION START: Apply Seaborn theme & Style Refinements <---
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6), dpi=300) # Keep size
    # ---> MODIFICATION END <---

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
    # Ensure min_fvu is slightly below the actual minimum for better axis range
    plot_min_fvu = min(min_fvu, 0.95) # Start axis slightly below lowest value or 0.95
    plot_min_fvu = max(0, (math.floor(plot_min_fvu * 20) / 20) - 0.05) # Round down to nearest 0.05, ensure >=0


    # Custom Y-axis scale only if data spans across 1.0
    if min_fvu < 1.0 and max_fvu > 1.0:
        transform = get_custom_scale_transform(max_fvu, plot_min_fvu, z=0.7)
        plt.gca().set_yscale(FuncScale(plt.gca().yaxis, transform))
        # Define ticks more dynamically
        below_100_ticks = np.arange(plot_min_fvu, 1.0, 0.05)
        # Sensible ticks above 100%
        upper_bound = math.ceil(max_fvu * 10) / 10 # Go slightly above max
        above_100_ticks = np.arange(1.0, upper_bound + 0.5 , 0.5) # Ticks every 50%
        yticks = np.unique(np.concatenate([below_100_ticks, above_100_ticks]))
        # Limit number of ticks shown if too dense
        if len(yticks) > 20:
            yticks = yticks[::max(1, len(yticks)//20)]
        plt.yticks(yticks)
        print(f"Using custom scale: min={plot_min_fvu:.2f}, max={max_fvu:.2f}")
    else:
        # Use linear scale if all data is >= 1 or <= 1
        plt.ylim(bottom=max(0, plot_min_fvu - 0.05), top=min(1.0, max_fvu + 0.05) if max_fvu <=1 else max_fvu + 0.05)
        print(f"Using linear scale: min={plot_min_fvu:.2f}, max={max_fvu:.2f}")


    # Define colors/styles (adjust as needed)
    # ---> MODIFICATION START: New Palette, Markers, Linewidths <---
    colors = {
        "assistant": "#9467bd", # Muted Purple (from tab10)
        "think": "#ff9e6d",     # Salmon/Orange
        "hint": "#69b3a2"      # Teal/Green
    }
    # Use solid lines for all, differentiate with markers
    linestyles = {"assistant": "-", "think": "-", "hint": "-"}
    markers = {"assistant": "o", "think": "^", "hint": "s"}
    linewidths = {"assistant": 2.0, "think": 2.0, "hint": 2.0}
    markersize = 4 # Small markers
    # ---> MODIFICATION END <---

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
                # Add markers
                marker=markers.get(pos_name, 'o'),
                markersize=markersize
            )

    # Use slightly lighter grid with seaborn style
    # ---> MODIFICATION START: Lighter Grid <---
    plt.grid(True, alpha=0.3, linestyle=':', color='#D3D3D3', linewidth=0.5)
    # ---> MODIFICATION END <---

    # Set x-ticks based on actual layers with data
    sorted_layers = sorted(list(layers_plotted))
    if sorted_layers:
        # Show every 4th label if many layers, else every 2nd
        step = 4 if n_layers > 40 else 2
        tick_layers = [l for l in sorted_layers if l % step == 0]
        plt.xticks(tick_layers, rotation=0, fontsize=10)
        plt.xlim(left=min(sorted_layers)-0.5, right=max(sorted_layers)+0.5) # Add padding
    else:
         plt.xticks([]) # No data, no ticks

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.yticks(fontsize=10)

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Fraction of Variance Unexplained (FVU)\n(lower is better)", fontsize=12)
    # Cleaner title
    plt.title(f"{model_name} ({n_layers} layers)\nFVU by Layer and Token Position", fontsize=13, pad=15)
    # Move legend outside plot area
    plt.legend(title="Token Position", title_fontsize='11', fontsize='10', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # ---> MODIFICATION START: Remove top/right spines <---
    sns.despine()
    # ---> MODIFICATION END <---

    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout more for legend
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight') # Use bbox_inches to include legend
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
    # ---> MODIFICATION START: More descriptive filename <--- 
    plot_filename = f"{MODEL_NAME_FOR_PATH}_{DS_NAME}_{HINT_TYPE}_{N_QUESTIONS_STR}_seed{SPLIT_SEED}_fvu.png"
    OUTPUT_PLOT_PATH = BASE_DIR / "analysis" / "plots" / plot_filename
    # ---> MODIFICATION END <---

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