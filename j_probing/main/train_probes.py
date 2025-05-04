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
from dataclasses import asdict
from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

# Import refactored components using absolute path
from j_probing.utils.training_utils import (
    DataConfig,
    ModelConfig,
    ProbeConfig,
    TrainingConfig,
    setup_determinism,
    load_target_data,
    load_question_ids,
    get_data_splits,
    load_activations_for_split,
    ActivationDataset,
    collate_fn,
    LinearProbe, # Keep LinearProbe import if needed directly
    train_probe,
)

# --- Main Execution Logic --- (Specific to Single Split)

def run_probe_training(cfg: TrainingConfig):
    """Loads data and runs training for all layers and positions using a single split."""

    # Create base output directory if it doesn't exist
    # Modified path to include "single_split"
    base_output_dir = cfg.data.output_dir / cfg.model.name / "single_split" / f"seed_{cfg.split_seed}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving probes and results to: {base_output_dir}")

    print("Loading data...")
    target_map = load_target_data(cfg.data.probing_data_path)
    all_qids_ordered = load_question_ids(cfg.data.meta_path)
    n_prompts = len(all_qids_ordered)
    print(f"Found {n_prompts} prompts in meta.json")
    print(f"Loaded targets for {len(target_map)} QIDs.")

    # Load model config from meta.json instead of TrainingConfig
    with open(cfg.data.meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    loaded_n_layers = meta_data.get("n_layers")
    loaded_d_model = meta_data.get("d_model")
    loaded_model_name = meta_data.get("model_name_from_config", cfg.model.name) # Fallback

    if loaded_n_layers is None or loaded_d_model is None:
        raise ValueError(f"meta.json at {cfg.data.meta_path} missing n_layers or d_model.")

    # Verify or update the TrainingConfig's model details
    if loaded_n_layers != cfg.model.n_layers:
        print(f"[Warning] n_layers mismatch! Meta: {loaded_n_layers}, Config: {cfg.model.n_layers}. Using value from meta.json.")
    if loaded_d_model != cfg.model.d_model:
        print(f"[Warning] d_model mismatch! Meta: {loaded_d_model}, Config: {cfg.model.d_model}. Using value from meta.json.")
    if loaded_model_name != cfg.model.name:
        print(f"[Warning] model_name mismatch! Meta: {loaded_model_name}, Config: {cfg.model.name}. Using name from meta.json.")

    # Update the main cfg object with loaded values
    cfg.model = ModelConfig(
        name=loaded_model_name,
        n_layers=loaded_n_layers,
        d_model=loaded_d_model
    )

    # Initialize WandB Run Once if enabled
    use_wandb = cfg.wandb_project is not None
    run = None
    if use_wandb:
        run_name = f"{cfg.model.name}_L{cfg.model.n_layers}_D{cfg.model.d_model}_seed{cfg.split_seed}"
        try:
            run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=run_name,
                config=asdict(cfg), # Log the entire config
                save_code=False
            )
            print(f"Wandb run initialized: {run.url}")
        except Exception as e:
            print(f"[Error] Failed to initialize Wandb: {e}. Disabling Wandb for this run.")
            use_wandb = False
            run = None

    train_qids, val_qids, test_qids = get_data_splits(
        all_qids_ordered, target_map, cfg.val_frac, cfg.test_frac, cfg.split_seed
    )

    # Prepare labels for each split
    train_labels = [target_map[qid] for qid in train_qids]
    val_labels = [target_map[qid] for qid in val_qids]
    test_labels = [target_map[qid] for qid in test_qids]

    position_map = {0: "assistant", 1: "think", 2: "hint"}
    all_results = {}

    # Load test activations once outside the loop (for efficiency)
    test_acts_all_pos = {} # Cache test activations for all positions for a layer

    # Use the corrected n_layers from cfg.model
    for layer in range(cfg.model.n_layers):
        layer_results = {}
        activation_path = cfg.data.activations_dir / f"layer_{layer:02d}.bin"
        if not activation_path.exists():
            print(f"[Warning] Activation file not found, skipping: {activation_path}")
            continue

        print(f"\n=== Processing Layer {layer} ===")
        # Clear previous layer's test activation cache
        test_acts_all_pos.clear()
        test_acts_loaded_for_layer = False

        for pos_idx, pos_name in position_map.items():
            print(f"  -- Position: {pos_name} (idx {pos_idx}) --")

            # --- Load Activations for this Layer/Position/Split ---
            print("    Loading activations...")
            try:
                train_acts = load_activations_for_split(
                    activation_path, pos_idx, all_qids_ordered, train_qids, cfg.model.d_model
                )
                val_acts = load_activations_for_split(
                    activation_path, pos_idx, all_qids_ordered, val_qids, cfg.model.d_model
                )
                # Load test activations only if not already cached for this layer/pos
                if pos_idx not in test_acts_all_pos:
                     test_acts_all_pos[pos_idx] = load_activations_for_split(
                          activation_path, pos_idx, all_qids_ordered, test_qids, cfg.model.d_model
                     )
                     test_acts_loaded_for_layer = True

                current_test_acts = test_acts_all_pos[pos_idx]

            except (FileNotFoundError, ValueError) as e:
                 print(f"    [Error] Failed to load activations for L{layer} P{pos_idx}: {e}. Skipping.")
                 continue # Skip this specific probe

            # --- Create Datasets and DataLoaders ---
            print("    Creating datasets/loaders...")
            train_dataset = ActivationDataset(train_acts, train_labels, cfg.device, cfg.dtype)
            val_dataset = ActivationDataset(val_acts, val_labels, cfg.device, cfg.dtype)
            test_dataset = ActivationDataset(current_test_acts, test_labels, cfg.device, cfg.dtype)

            # Use full dataset batch size for val/test for faster evaluation
            train_loader = DataLoader(train_dataset, batch_size=cfg.probe_config.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset) if len(val_dataset)>0 else 1, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset) if len(test_dataset)>0 else 1, shuffle=False, collate_fn=collate_fn)

            # --- Configure and Train Probe --- 
            current_probe_cfg = ProbeConfig(
                layer=layer,
                position_index=pos_idx,
                position_name=pos_name,
                lr=cfg.probe_config.lr,
                beta1=cfg.probe_config.beta1,
                beta2=cfg.probe_config.beta2,
                batch_size=cfg.probe_config.batch_size,
                weight_init_range=cfg.probe_config.weight_init_range,
                weight_init_seed=cfg.probe_config.weight_init_seed # Use the seed from config
            )

            best_state_dict, final_metrics = train_probe(
                probe_config=current_probe_cfg,
                d_model=cfg.model.d_model,
                device=cfg.device,
                # dtype=cfg.dtype, # dtype passed implicitly via device/data
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                # Pass training loop params from main config
                max_steps=cfg.max_steps,
                patience=cfg.patience,
                val_freq=cfg.val_freq,
                use_wandb=use_wandb,
                wandb_log_prefix="" # No prefix for single split
            )

            # --- Save probe weights and metrics --- 
            if best_state_dict:
                probe_output_dir = base_output_dir / f"layer_{layer:02d}" / pos_name
                probe_output_dir.mkdir(parents=True, exist_ok=True)

                # Save weights
                torch.save(best_state_dict, probe_output_dir / "probe_weights.pt")

                # Save metrics
                with open(probe_output_dir / "metrics.json", "w") as f:
                    json.dump(final_metrics, f, indent=2)

                layer_results[pos_name] = final_metrics
            else:
                print("    Skipping saving due to no valid best state.")
                layer_results[pos_name] = {"error": "Training failed or stopped early without improvement."}

        all_results[f"layer_{layer}"] = layer_results
        # Clear cached test activations for the layer if they were loaded
        if test_acts_loaded_for_layer:
             test_acts_all_pos.clear()

    # --- Aggregate and Save Final Results Summary --- 
    final_results_path = base_output_dir / "all_results_summary.json"
    print(f"\n=== Training Complete ===")
    print(f"Saving summary results to {final_results_path}")
    with open(final_results_path, "w") as f:
         json.dump(all_results, f, indent=2)

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

    # Default probe hyperparameters (can be overridden)
    default_probe_config = ProbeConfig(
        layer=-1, # Placeholder
        position_index=-1, # Placeholder
        position_name="placeholder", # Placeholder
        batch_size=64,
        weight_init_seed=42
        # Other ProbeConfig params use defaults
    )

    config = TrainingConfig(
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
        split_seed=42, # Seed for the single train/val/test split
        # wandb_project=None, # Set to None to disable WandB
        # wandb_entity="your_entity", # Your WandB entity
    )

    run_probe_training(config) 