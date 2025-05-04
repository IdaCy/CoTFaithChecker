#!/usr/bin/env python3
import os
import sys
print(f"Running script from CWD: {os.getcwd()}")
print(f"Python Path: {sys.path}")

# Add project root to sys.path to allow absolute imports
from pathlib import Path
project_root = Path(__file__).resolve().parents[2] # CoTFaithChecker directory
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Updated Python Path: {sys.path}")

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import collections

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
import wandb

# Import refactored components using absolute path
from j_probing.utils.training_utils import (
    DataConfig,
    ModelConfig,
    ProbeConfig,
    # TrainingConfig, # Use CVConfig instead
    setup_determinism,
    load_target_data,
    load_question_ids,
    # get_data_splits, # Use KFold instead
    load_activations_for_split,
    ActivationDataset,
    collate_fn,
    LinearProbe,
    train_probe,
)

# --- Configuration for Cross-Validation ---

@dataclass
class CVConfig:
    """Overall configuration for K-Fold Cross-Validation training."""
    data: DataConfig
    model: ModelConfig
    probe_config: ProbeConfig # Base probe hyperparameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 # Activation dtype
    # --- CV Specific --- 
    k_folds: int = 5 # Number of folds
    cv_split_seed: int = 42 # Seed for KFold shuffling
    # --- Validation within Fold --- 
    internal_val_frac: float = 0.15 # Fraction of training fold used for validation
    internal_split_seed: int = 43 # Separate seed for val split within fold
    # --- Training Loop Params (used by train_probe) ---
    max_steps: int = 5000
    patience: int = 500
    val_freq: int = 50
    # --- Logging ---
    wandb_project: Optional[str] = "faithfulness_probes_cv" # Different project potentially
    wandb_entity: Optional[str] = "officer_k"

# --- Main K-Fold Execution Logic ---

def run_cv_probe_training(cfg: CVConfig):
    """Loads data and runs K-Fold CV training for all layers and positions."""

    # Create base output directory for CV results
    base_output_dir = cfg.data.output_dir / cfg.model.name / f"cv_k{cfg.k_folds}" / f"seed_{cfg.cv_split_seed}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving CV probes and results to: {base_output_dir}")

    print("Loading data...")
    target_map = load_target_data(cfg.data.probing_data_path)
    all_qids_ordered = load_question_ids(cfg.data.meta_path)
    n_prompts = len(all_qids_ordered)
    print(f"Found {n_prompts} prompts in meta.json")

    # Filter QIDs to those present in target_map
    valid_qids = [qid for qid in all_qids_ordered if qid in target_map]
    valid_qids_np = np.array(valid_qids) # Use numpy array for easier indexing
    print(f"Using {len(valid_qids)} QIDs with valid targets for CV.")

    # Load model config from meta.json
    with open(cfg.data.meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    loaded_n_layers = meta_data.get("n_layers")
    loaded_d_model = meta_data.get("d_model")
    loaded_model_name = meta_data.get("model_name_from_config", cfg.model.name)

    if loaded_n_layers is None or loaded_d_model is None:
        raise ValueError(f"meta.json missing n_layers or d_model.")

    # Update cfg.model with loaded values
    cfg.model = ModelConfig(name=loaded_model_name, n_layers=loaded_n_layers, d_model=loaded_d_model)

    # Initialize WandB Run Once if enabled
    use_wandb = cfg.wandb_project is not None
    run = None
    if use_wandb:
        run_name = f"{cfg.model.name}_CV{cfg.k_folds}_L{cfg.model.n_layers}_D{cfg.model.d_model}_seed{cfg.cv_split_seed}"
        try:
            run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=run_name,
                config=asdict(cfg), # Log the CV config
                save_code=False
            )
            print(f"Wandb run initialized: {run.url}")
        except Exception as e:
            print(f"[Error] Failed to initialize Wandb: {e}. Disabling Wandb.")
            use_wandb = False
            run = None

    # Initialize KFold
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.cv_split_seed)

    position_map = {0: "assistant", 1: "think", 2: "hint"}
    all_fold_results = collections.defaultdict(lambda: collections.defaultdict(list))
    # Structure: all_fold_results[layer_pos_key][metric_name] = [fold1_val, fold2_val, ...]

    # --- Outer Loop: Layers and Positions ---
    for layer in range(cfg.model.n_layers):
        activation_path = cfg.data.activations_dir / f"layer_{layer:02d}.bin"
        if not activation_path.exists():
            print(f"[Warning] Act file not found, skipping layer {layer}: {activation_path}")
            continue

        print(f"\n=== Processing Layer {layer} ===")
        for pos_idx, pos_name in position_map.items():
            layer_pos_key = f"L{layer}_{pos_name}"
            print(f"  -- Position: {pos_name} (idx {pos_idx}) --")

            fold_metrics_accumulator = collections.defaultdict(list)
            # Structure: { "test_loss": [fold1_loss, ...], "test_pearson_r": [fold1_r, ...] }

            # --- Inner Loop: Folds ---
            for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(valid_qids_np)):
                print(f"    -- Fold {fold_idx + 1}/{cfg.k_folds} --")

                # Get QIDs for this fold's splits
                train_val_qids = list(valid_qids_np[train_val_indices])
                test_qids = list(valid_qids_np[test_indices])

                # Internal split for validation (using train_val_qids)
                if cfg.internal_val_frac > 0 and len(train_val_qids) > 1:
                    setup_determinism(cfg.internal_split_seed + fold_idx) # Vary seed per fold
                    try:
                        train_qids, val_qids = train_test_split(
                            train_val_qids, test_size=cfg.internal_val_frac, shuffle=True
                        )
                    except ValueError: # Handle case where split is not possible (too few samples)
                         print(f"[Warning] Could not perform internal validation split for fold {fold_idx+1}. Using all for training.")
                         train_qids = train_val_qids
                         val_qids = []
                else:
                    train_qids = train_val_qids
                    val_qids = []

                # Prepare labels for the fold splits
                train_labels = [target_map[qid] for qid in train_qids]
                val_labels = [target_map[qid] for qid in val_qids]
                test_labels = [target_map[qid] for qid in test_qids]

                # Load activations for this fold
                print(f"      Loading activations for fold {fold_idx + 1}...")
                try:
                    train_acts = load_activations_for_split(activation_path, pos_idx, all_qids_ordered, train_qids, cfg.model.d_model)
                    val_acts = load_activations_for_split(activation_path, pos_idx, all_qids_ordered, val_qids, cfg.model.d_model) if val_qids else np.array([])
                    test_acts = load_activations_for_split(activation_path, pos_idx, all_qids_ordered, test_qids, cfg.model.d_model)
                except (FileNotFoundError, ValueError) as e:
                    print(f"      [Error] Failed to load activations for L{layer} P{pos_idx} Fold {fold_idx+1}: {e}. Skipping fold.")
                    # Record NaN or skip fold? Let's skip and note it.
                    fold_metrics_accumulator["test_loss"].append(float('nan'))
                    fold_metrics_accumulator["test_pearson_r"].append(float('nan'))
                    continue # Skip to next fold

                # Create datasets and dataloaders for the fold
                train_dataset = ActivationDataset(train_acts, train_labels, cfg.device, cfg.dtype)
                val_dataset = ActivationDataset(val_acts, val_labels, cfg.device, cfg.dtype) if len(val_qids) > 0 else None
                test_dataset = ActivationDataset(test_acts, test_labels, cfg.device, cfg.dtype)

                train_loader = DataLoader(train_dataset, batch_size=cfg.probe_config.batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=len(val_dataset) if val_dataset else 1, shuffle=False, collate_fn=collate_fn) if val_dataset else None
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) if len(test_dataset) > 0 else 1, shuffle=False, collate_fn=collate_fn)

                # Configure and Train Probe for this fold
                # Ensure weight init seed varies per fold if desired, or keep fixed
                fold_probe_seed = cfg.probe_config.weight_init_seed # Fixed across folds by default
                # fold_probe_seed = cfg.probe_config.weight_init_seed + fold_idx # Varies across folds

                current_probe_cfg = ProbeConfig(
                    layer=layer,
                    position_index=pos_idx,
                    position_name=pos_name,
                    lr=cfg.probe_config.lr,
                    beta1=cfg.probe_config.beta1,
                    beta2=cfg.probe_config.beta2,
                    batch_size=cfg.probe_config.batch_size,
                    weight_init_range=cfg.probe_config.weight_init_range,
                    weight_init_seed=fold_probe_seed # Seed for this fold's probe init
                )

                best_state_dict, final_metrics = train_probe(
                    probe_config=current_probe_cfg,
                    d_model=cfg.model.d_model,
                    device=cfg.device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    max_steps=cfg.max_steps,
                    patience=cfg.patience,
                    val_freq=cfg.val_freq,
                    use_wandb=use_wandb,
                    wandb_log_prefix=f"Fold{fold_idx+1}/" # Add fold prefix for step logs
                )

                # Store fold's test metrics
                test_loss = final_metrics.get("test_loss")
                test_pearson = final_metrics.get("test_pearson_r")
                fold_metrics_accumulator["test_loss"].append(test_loss if test_loss is not None else float('nan'))
                fold_metrics_accumulator["test_pearson_r"].append(test_pearson if test_pearson is not None else float('nan'))

                # Optional: Save individual fold probes/metrics if needed
                # fold_output_dir = base_output_dir / f"layer_{layer:02d}" / pos_name / f"fold_{fold_idx+1}"
                # fold_output_dir.mkdir(parents=True, exist_ok=True)
                # if best_state_dict: torch.save(best_state_dict, fold_output_dir / "probe_weights.pt")
                # with open(fold_output_dir / "metrics.json", "w") as f: json.dump(final_metrics, f, indent=2)

            # --- Aggregation after Folds for Layer/Position ---
            aggregated_metrics = {}
            for metric, values in fold_metrics_accumulator.items():
                valid_values = [v for v in values if not math.isnan(v)]
                if valid_values:
                    aggregated_metrics[f"{metric}_mean"] = np.mean(valid_values)
                    aggregated_metrics[f"{metric}_std"] = np.std(valid_values)
                    aggregated_metrics[f"{metric}_values"] = valid_values # Keep individual values
                else:
                    aggregated_metrics[f"{metric}_mean"] = float('nan')
                    aggregated_metrics[f"{metric}_std"] = float('nan')
                    aggregated_metrics[f"{metric}_values"] = []

            print(f"    Aggregated Test Loss: {aggregated_metrics.get('test_loss_mean'):.4f} +/- {aggregated_metrics.get('test_loss_std'):.4f}")
            print(f"    Aggregated Test Pearson R: {aggregated_metrics.get('test_pearson_r_mean'):.4f} +/- {aggregated_metrics.get('test_pearson_r_std'):.4f}")

            # Store aggregated results
            all_fold_results[layer_pos_key] = aggregated_metrics

            # Log aggregated metrics to WandB
            if use_wandb:
                wandb_agg_log = {}
                for metric, value in aggregated_metrics.items():
                    if "_values" not in metric: # Don't log the list of values
                        wandb_agg_log[f"aggregated/{layer_pos_key}/{metric}"] = value
                wandb.log(wandb_agg_log)

    # --- Save Final Aggregated Results Summary ---
    final_results_path = base_output_dir / "all_cv_results_summary.json"
    print(f"\n=== CV Training Complete ===")
    print(f"Saving summary results to {final_results_path}")
    # Convert numpy types for JSON serialization if necessary
    serializable_results = {}
    for key, metrics in all_fold_results.items():
        serializable_metrics = {}
        for m_key, m_val in metrics.items():
             if isinstance(m_val, np.generic):
                 serializable_metrics[m_key] = m_val.item() # Convert numpy types
             else:
                 serializable_metrics[m_key] = m_val
        serializable_results[key] = serializable_metrics

    with open(final_results_path, "w") as f:
         json.dump(serializable_results, f, indent=2)

    # Finish WandB Run if it was initialized
    if use_wandb and run:
        run.finish()
        print("Wandb run finished.")

if __name__ == "__main__":
    # --- Define Experiment Configuration ---
    DS_NAME = "mmlu"
    MODEL_NAME_FOR_PATH = "DeepSeek-R1-Distill-Llama-8B"
    HINT_TYPE = "sycophancy"
    N_QUESTIONS_STR = "5001"

    BASE_DIR = Path("j_probing")
    DATA_DIR = BASE_DIR / "data" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    ACTS_DIR = BASE_DIR / "acts" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    PROBE_OUTPUT_DIR = BASE_DIR / "probes"

    # Default probe hyperparameters
    default_probe_config = ProbeConfig(
        layer=-1, position_index=-1, position_name="placeholder", # Placeholders
        batch_size=64,
        weight_init_seed=42 # Seed for probe weight init (fixed across folds by default)
    )

    # CV Configuration
    cv_config = CVConfig(
        data=DataConfig(
            probing_data_path=DATA_DIR / "probing_data.json",
            activations_dir=ACTS_DIR,
            meta_path=ACTS_DIR / "meta.json",
            output_dir=PROBE_OUTPUT_DIR
        ),
        model=ModelConfig( # Placeholders, loaded from meta.json
            name=MODEL_NAME_FOR_PATH,
            d_model=-1,
            n_layers=-1
        ),
        probe_config=default_probe_config,
        k_folds=5, # Number of folds
        cv_split_seed=42, # Seed for the KFold split itself
        internal_val_frac=0.15, # Use 15% of train fold for validation
        internal_split_seed=43, # Seed for the internal val split
        # wandb_project="faithfulness_probes_cv", # WandB project name
        # wandb_entity="your_entity", # Your WandB entity
    )

    run_cv_probe_training(cv_config) 