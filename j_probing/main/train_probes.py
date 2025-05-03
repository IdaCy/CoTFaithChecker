#!/usr/bin/env python3
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb  # Optional, but good for tracking
from scipy.stats import pearsonr

# Assuming a utility function like this exists or is adapted
# If not, remove or replace the setup_determinism calls
def setup_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Potentially slow down training, but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- Configuration Dataclasses ---

@dataclass(frozen=True)
class DataConfig:
    """Configuration related to dataset paths and properties."""
    probing_data_path: Path
    activations_dir: Path
    meta_path: Path
    output_dir: Path = Path("j_probing/probes") # Default output dir

@dataclass(frozen=True)
class ModelConfig:
    """Configuration related to the model whose activations are probed."""
    name: str
    d_model: int
    n_layers: int

@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for a single probe's training."""
    layer: int
    position_index: int # 0: assistant, 1: think, 2: hint
    position_name: str  # Descriptive name for logging
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int = 32 # Adjusted default, can be overridden
    weight_init_range: float = 0.02
    weight_init_seed: int = 42 # Seed for initializing probe weights

@dataclass
class TrainingConfig: # Made mutable to allow setting probe_config later
    """Overall configuration for the training run."""
    data: DataConfig
    model: ModelConfig
    probe_config: ProbeConfig # Include default probe config here
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 # Dtype for loading activations
    # --- Splitting ---
    val_frac: float = 0.10
    test_frac: float = 0.10
    split_seed: int = 42 # Seed for train/val/test split
    # --- Training Loop ---
    max_steps: int = 5000
    patience: int = 500 # Steps to wait for val loss improvement
    val_freq: int = 50  # Steps between validation checks
    # --- Logging ---
    wandb_project: Optional[str] = "faithfulness_probes" # Set to None to disable wandb
    wandb_entity: Optional[str] = "officer_k" # Set to your wandb entity

@dataclass
class CollateFnOutput:
    """Output structure for the DataLoader collate function."""
    activations: torch.Tensor # Float[torch.Tensor, "batch d_model"]
    labels: torch.Tensor # Float[torch.Tensor, "batch"]

# --- Data Loading & Handling ---

def load_target_data(path: Path) -> Dict[int, float]:
    """Loads probing_data.json and returns a map from question_id to prob_verb_match."""
    target_map = {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for record in data:
        qid = record["question_id"]
        prob = record.get("prob_verb_match") # Use .get in case key is missing
        if prob is not None: # Filter out null probabilities
             target_map[qid] = float(prob)
        else:
            print(f"[Warning] Skipping QID {qid}: 'prob_verb_match' is null or missing.")
    if not target_map:
        raise ValueError(f"No valid 'prob_verb_match' values found in {path}")
    return target_map

def load_question_ids(meta_path: Path) -> List[int]:
    """Loads the meta.json file and returns the ordered list of question IDs."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    qids = meta_data.get("question_ids")
    if qids is None:
        raise ValueError(f"'question_ids' not found in {meta_path}")
    if not isinstance(qids, list):
         raise ValueError(f"'question_ids' in {meta_path} is not a list")
    return qids

def get_data_splits(
    all_qids: List[int],
    target_map: Dict[int, float],
    val_frac: float,
    test_frac: float,
    seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """Splits question IDs into train, validation, and test sets randomly."""
    # Filter qids to only include those present in the target map
    valid_qids = [qid for qid in all_qids if qid in target_map]
    if len(valid_qids) != len(all_qids):
        print(f"[Warning] {len(all_qids) - len(valid_qids)} QIDs were in meta.json but missing from target data (or had null prob_verb_match).")

    n_total = len(valid_qids)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_test - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Dataset size ({n_total}) too small for specified splits (train: {n_train}, val: {n_val}, test: {n_test}).")

    setup_determinism(seed) # Ensure reproducible splits
    shuffled_qids = random.sample(valid_qids, n_total)

    test_qids = shuffled_qids[:n_test]
    val_qids = shuffled_qids[n_test : n_test + n_val]
    train_qids = shuffled_qids[n_test + n_val :]

    print(f"Data split (seed {seed}): {len(train_qids)} train, {len(val_qids)} val, {len(test_qids)} test")
    return train_qids, val_qids, test_qids


def load_activations_for_split(
    activation_path: Path,
    position_index: int,
    qids_in_order: List[int],
    split_qids: List[int],
    d_model: int,
    # dtype: torch.dtype = torch.float16 # dtype is for torch, memmap uses numpy
) -> np.ndarray:
    """Loads a slice of activations for a specific split using memmap."""
    n_total = len(qids_in_order)
    # Important: Need to know the *original* dtype of the memmap file
    # Assuming float16 based on get_acts.py default
    memmap_dtype = np.float16

    # Map question IDs to their row index in the memmap file
    qid_to_row_index = {qid: idx for idx, qid in enumerate(qids_in_order)}

    # Get the row indices corresponding to the current split
    row_indices = [qid_to_row_index[qid] for qid in split_qids if qid in qid_to_row_index]
    if len(row_indices) != len(split_qids):
         print(f"[Warning] Mismatch between split QIDs ({len(split_qids)}) and QIDs found in meta.json ({len(row_indices)}) for {activation_path}")

    if not row_indices:
        # Return an empty array with the correct dimensions if the split is empty
        # (though get_data_splits should prevent this)
        return np.empty((0, d_model), dtype=memmap_dtype)

    # Load the memmap file
    try:
        memmap = np.memmap(
            filename=str(activation_path),
            mode="r", # Read-only
            dtype=memmap_dtype,
            shape=(n_total, d_model, 3), # N, D, 3 (positions)
        )
    except FileNotFoundError:
        print(f"[Error] Activation file not found: {activation_path}")
        raise
    except ValueError as e:
        # Check if shape mismatch is the likely cause
        expected_size = n_total * d_model * 3 * np.dtype(memmap_dtype).itemsize
        actual_size = activation_path.stat().st_size
        print(f"[Error] Loading {activation_path}. Expected size {expected_size}, actual size {actual_size}.")
        print(f"Shape mismatch or other error loading: {e}")
        raise

    # Extract the specific position and the rows for the current split
    # Directly index the rows needed, then select the position slice
    # This avoids loading unnecessary rows into memory before slicing
    split_activations = memmap[row_indices, :, position_index]

    # It's generally safer to copy data out of a memmap before extensive use
    # especially if converting dtype or moving to torch tensor later.
    return np.array(split_activations) # Creates a copy in memory


class ActivationDataset(Dataset):
    """Simple PyTorch Dataset for activations and labels."""
    def __init__(
        self,
        activations: np.ndarray,
        labels: List[float],
        device: str,
        dtype: torch.dtype = torch.float16 # Activation dtype
    ):
        if activations.shape[0] != len(labels):
            raise ValueError(f"Mismatch between number of activations ({activations.shape[0]}) and labels ({len(labels)})")
        # Move data to the target device and type during initialization
        # Activations can stay in float16/bfloat16
        self.activations = torch.from_numpy(activations).to(dtype=dtype, device=device)
        # Labels/Loss usually float32 for stability
        self.labels = torch.tensor(labels, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: # Tuple[Float[torch.Tensor, "d_model"], Float[torch.Tensor, ""]]:
        return self.activations[idx], self.labels[idx]

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> CollateFnOutput:
    """Collate function to batch activations and labels."""
    activations = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return CollateFnOutput(activations=activations, labels=labels)


# --- Probe Model ---

class LinearProbe(nn.Module):
    """Simple linear probe with bias."""
    def __init__(self, d_model: int, init_range: float = 0.02, seed: int = 42):
        super().__init__()
        setup_determinism(seed) # For reproducible weight initialization
        # Initialize weights and bias in float32 for stability
        self.w = nn.Parameter(torch.randn(d_model, dtype=torch.float32) * init_range)
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        # Both parameters should be on the same device
        return self.w.device

    def forward(
        self,
        activations: torch.Tensor # Float[torch.Tensor, "batch d_model"]
    ) -> torch.Tensor: # Float[torch.Tensor, "batch"]:
        # Cast activations to float32 for matmul stability if needed
        acts_float32 = activations.to(torch.float32)
        # Ensure bias is broadcast correctly: (batch,)
        return (acts_float32 @ self.w) + self.b.squeeze()


# --- Training & Evaluation Logic ---

def evaluate_probe(
    probe: LinearProbe,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    dtype: torch.dtype # Probe's internal dtype (float32)
    ) -> Tuple[float, float]:
    """Evaluates the probe on a dataset, returns MSE loss and Pearson correlation."""
    probe.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            activations = batch.activations.to(device=device) # Ensure data on correct device
            labels = batch.labels.to(device=device)

            predictions = probe(activations) # Forward pass uses float32 internally

            # Ensure predictions and labels are compatible for loss calculation
            # Predictions should be (batch_size,), labels should be (batch_size,)
            loss = criterion(predictions, labels)

            if not math.isnan(loss.item()):
                 total_loss += loss.item()
                 num_batches += 1
                 all_preds.extend(predictions.cpu().numpy())
                 all_labels.extend(labels.cpu().numpy())
            else:
                 print("[Warning] NaN loss detected during evaluation, skipping batch.")


    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')

    # Calculate Pearson correlation
    if len(all_labels) > 1 and len(all_preds) == len(all_labels) and np.std(all_labels) > 1e-6 and np.std(all_preds) > 1e-6:
         correlation, _ = pearsonr(all_labels, all_preds)
         if math.isnan(correlation):
              correlation = 0.0 # Handle potential NaN if variance is zero
    else:
         correlation = 0.0 # Not enough data or zero variance

    probe.train() # Set back to train mode
    return avg_loss, correlation


def train_probe(
    probe_config: ProbeConfig,
    d_model: int,
    device: str,
    dtype: torch.dtype,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None, # Add test_loader for final eval
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Trains a single linear probe and returns best state_dict and metrics."""

    run_name = f"L{probe_config.layer}_{probe_config.position_name}"
    use_wandb = False
    run = None

    print(f"--- Training probe: Layer {probe_config.layer}, Pos {probe_config.position_name} ---")

    # --- Initialization ---
    probe = LinearProbe(
        d_model=d_model,
        init_range=probe_config.weight_init_range,
        seed=probe_config.weight_init_seed
    ).to(device) # Initial device placement

    optimizer = Adam(
        probe.parameters(), lr=probe_config.lr, betas=(probe_config.beta1, probe_config.beta2)
    )
    criterion = nn.MSELoss()

    # --- Training Loop ---
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    steps_done = 0

    train_iter = iter(train_loader)
    probe.train()

    max_steps = 5000
    patience = 500
    val_freq = 50

    pbar = tqdm(total=max_steps, desc=f"Train L{probe_config.layer} P{probe_config.position_index}", leave=False)

    while steps_done < max_steps:
        try:
            batch = next(train_iter)
            batch_acts = batch.activations.to(device) # Ensure data on correct device
            batch_labels = batch.labels.to(device)
        except StopIteration:
            train_iter = iter(train_loader)
            continue

        optimizer.zero_grad()
        predictions = probe(batch_acts) # Forward pass uses float32
        loss = criterion(predictions, batch_labels)

        if math.isnan(loss.item()):
            print(f"[Warning] NaN loss detected at step {steps_done}. Skipping step.")
            steps_done += 1
            pbar.update(1)
            continue

        loss.backward()
        optimizer.step()
        steps_done += 1
        pbar.update(1)

        # Log training loss periodically
        if use_wandb and steps_done % val_freq == 0:
             # ---> MODIFICATION START: Adjust log keys <---
             log_data = {
                 f"train_loss_step/L{probe_config.layer}_{probe_config.position_name}": loss.item(),
                 "step": steps_done # Step is global across probes
             }
             wandb.log(log_data)
             # ---> MODIFICATION END <---


        # --- Validation ---
        if steps_done % val_freq == 0:
            val_loss, val_corr = evaluate_probe(probe, val_loader, criterion, device, torch.float32)
            if math.isnan(val_loss):
                print(f"[Warning] NaN validation loss at step {steps_done}. Stopping training for this probe.")
                break # Stop training if validation loss is NaN

            if use_wandb:
                 # ---> MODIFICATION START: Adjust log keys <---
                 log_data = {
                     f"val_loss/L{probe_config.layer}_{probe_config.position_name}": val_loss,
                     f"val_pearson_r/L{probe_config.layer}_{probe_config.position_name}": val_corr,
                     "step": steps_done # Step is global across probes
                 }
                 wandb.log(log_data)
                 # ---> MODIFICATION END <---

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Detach state_dict items before deepcopying
                best_model_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += val_freq
                if patience_counter >= patience:
                    print(f"Step {steps_done}: Early stopping triggered.")
                    break

    pbar.close()
    if best_model_state is None:
        print("[Warning] No improvement found during training. Using final model state.")
        best_model_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    # --- Final Evaluation on Test Set ---
    final_metrics = {"best_val_loss": best_val_loss, "steps_completed": steps_done}
    if test_loader is not None and best_model_state is not None:
        print("  Evaluating on test set...")
        # Load best model state
        probe.load_state_dict(best_model_state)
        test_loss, test_corr = evaluate_probe(probe, test_loader, criterion, device, torch.float32)
        # ---> MODIFICATION START: Cast results to standard float <---
        final_metrics["test_loss"] = float(test_loss) if not math.isnan(test_loss) else None
        final_metrics["test_pearson_r"] = float(test_corr) if not math.isnan(test_corr) else None
        # ---> MODIFICATION END <---
        if use_wandb:
             # ---> MODIFICATION START: Log final test metrics <---
             wandb.log({
                 f"final_test_loss/L{probe_config.layer}_{probe_config.position_name}": final_metrics["test_loss"],
                 f"final_test_pearson_r/L{probe_config.layer}_{probe_config.position_name}": final_metrics["test_pearson_r"]
                 # Log these once at the end for this specific probe
             })
             # ---> MODIFICATION END <---
        print(f"  Test Loss: {test_loss:.4f}, Test Pearson R: {test_corr:.4f}")
    else:
         print("  Skipping test set evaluation.")
         final_metrics["test_loss"] = None
         final_metrics["test_pearson_r"] = None


    print(f"--- Finished training: Layer {probe_config.layer}, Pos {probe_config.position_name} ---")
    print(f"Final Metrics: {final_metrics}")

    # ---> MODIFICATION START: Remove internal wandb.finish <---
    # if use_wandb and run:
    #     run.finish() # Finish the wandb run
    # ---> MODIFICATION END <---

    return best_model_state, final_metrics


# --- Main Execution Logic ---

def run_probe_training(cfg: TrainingConfig):
    """Loads data and runs training for all layers and positions."""

    # Create base output directory if it doesn't exist
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

    # Use the loaded values for the rest of the process
    # Store the correct values back into the main cfg object for simplicity downstream
    cfg.model = ModelConfig( 
        name=loaded_model_name,
        n_layers=loaded_n_layers,
        d_model=loaded_d_model
    )

    # ---> MODIFICATION START: Initialize WandB Run Once <---
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
    # ---> MODIFICATION END <---

    train_qids, val_qids, test_qids = get_data_splits(
        all_qids_ordered, target_map, cfg.val_frac, cfg.test_frac, cfg.split_seed
    )

    # Prepare labels for each split
    train_labels = [target_map[qid] for qid in train_qids]
    val_labels = [target_map[qid] for qid in val_qids]
    test_labels = [target_map[qid] for qid in test_qids]


    position_map = {0: "assistant", 1: "think", 2: "hint"}

    all_results = {}
    # Load test activations once outside the loop
    test_acts_all_layers = {}


    # Use the corrected n_layers from cfg.model
    for layer in range(cfg.model.n_layers):
        layer_results = {}
        activation_path = cfg.data.activations_dir / f"layer_{layer:02d}.bin"
        if not activation_path.exists():
            print(f"[Warning] Activation file not found, skipping: {activation_path}")
            continue

        print(f"\n=== Processing Layer {layer} ===")
        test_acts_layer_loaded = False


        for pos_idx, pos_name in position_map.items():
            print(f"  -- Position: {pos_name} (idx {pos_idx}) --")

            # --- Load Activations for this Layer/Position/Split ---
            print("    Loading activations...")
            try:
                # Use the corrected d_model from cfg.model
                train_acts = load_activations_for_split(
                    activation_path, pos_idx, all_qids_ordered, train_qids, cfg.model.d_model
                )
                val_acts = load_activations_for_split(
                    activation_path, pos_idx, all_qids_ordered, val_qids, cfg.model.d_model
                )
                # Load test activations only once per layer if needed
                if layer not in test_acts_all_layers:
                     test_acts_all_layers[layer] = {}
                if pos_idx not in test_acts_all_layers[layer]:
                     test_acts_all_layers[layer][pos_idx] = load_activations_for_split(
                          activation_path, pos_idx, all_qids_ordered, test_qids, cfg.model.d_model
                     )
                     test_acts_layer_loaded = True # Flag that test acts were loaded for this layer

                current_test_acts = test_acts_all_layers[layer][pos_idx]

            except (FileNotFoundError, ValueError) as e:
                 print(f"    [Error] Failed to load activations for L{layer} P{pos_idx}: {e}. Skipping.")
                 continue # Skip this specific probe

            # --- Create Datasets and DataLoaders ---
            print("    Creating datasets/loaders...")
            train_dataset = ActivationDataset(train_acts, train_labels, cfg.device, cfg.dtype)
            val_dataset = ActivationDataset(val_acts, val_labels, cfg.device, cfg.dtype)
            test_dataset = ActivationDataset(current_test_acts, test_labels, cfg.device, cfg.dtype)


            # Batch size for val/test is full dataset for faster evaluation
            train_loader = DataLoader(train_dataset, batch_size=cfg.probe_config.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn) # Use full batch for test


            # --- Configure and Train Probe ---
            # Create a specific ProbeConfig for this run
            current_probe_cfg = ProbeConfig(
                layer=layer,
                position_index=pos_idx,
                position_name=pos_name,
                lr = cfg.probe_config.lr,
                beta1 = cfg.probe_config.beta1,
                beta2 = cfg.probe_config.beta2,
                batch_size = cfg.probe_config.batch_size,
                weight_init_range = cfg.probe_config.weight_init_range,
                weight_init_seed = cfg.probe_config.weight_init_seed
            )

            # ---> MODIFICATION START: Pass necessary parameters directly <---
            best_state_dict, final_metrics = train_probe(
                probe_config=current_probe_cfg,
                # Pass loaded/verified model and training params
                d_model=cfg.model.d_model,
                device=cfg.device,
                dtype=cfg.dtype, # Pass activation dtype
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader
                # TODO: Consider passing max_steps, patience, val_freq if they might change per probe
                # or keep them hardcoded/defaulted inside train_probe for now.
            )
            # ---> MODIFICATION END <---

            # --- Save probe weights and metrics ---
            if best_state_dict:
                probe_output_dir = base_output_dir / f"layer_{layer:02d}" / pos_name
                probe_output_dir.mkdir(parents=True, exist_ok=True)

                # Save weights
                torch.save(best_state_dict, probe_output_dir / "probe_weights.pt")

                # Save metrics
                with open(probe_output_dir / "metrics.json", "w") as f:
                    json.dump(final_metrics, f, indent=2)

                layer_results[pos_name] = final_metrics # Store metrics
            else:
                print("    Skipping saving due to no valid best state.")
                layer_results[pos_name] = {"error": "Training failed or stopped early without improvement."}


        all_results[f"layer_{layer}"] = layer_results
        # Clear cached test activations for the layer if they were loaded
        if test_acts_layer_loaded and layer in test_acts_all_layers:
             del test_acts_all_layers[layer]


    # --- Aggregate and Save Final Results ---
    final_results_path = base_output_dir / "all_results_summary.json"
    print(f"\n=== Training Complete ===")
    print(f"Saving summary results to {final_results_path}")
    with open(final_results_path, "w") as f:
         json.dump(all_results, f, indent=2)

    # ---> MODIFICATION START: Finish WandB Run <---
    if use_wandb and run:
        run.finish()
        print("Wandb run finished.")
    # ---> MODIFICATION END <---



if __name__ == "__main__":
    # --- Define Experiment Configuration ---
    # Keep these for path construction
    DS_NAME = "mmlu"
    MODEL_NAME_FOR_PATH = "DeepSeek-R1-Distill-Llama-8B" # Matches output dir name
    HINT_TYPE = "sycophancy"
    N_QUESTIONS_STR = "5001" # Adjust to your actual activation dir name

    # Construct paths using the variables
    BASE_DIR = Path("j_probing")
    DATA_DIR = BASE_DIR / "data" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    ACTS_DIR = BASE_DIR / "acts" / DS_NAME / MODEL_NAME_FOR_PATH / HINT_TYPE / N_QUESTIONS_STR
    PROBE_OUTPUT_DIR = BASE_DIR / "probes" # Define base output dir

    # --- Model Details are now LOADED from meta.json, not hardcoded here ---
    # MODEL_D_MODEL = 4096
    # MODEL_N_LAYERS = 32

    # Define default hyperparameters for the probe config part of TrainingConfig
    default_probe_config = ProbeConfig(
        layer=-1, # Placeholder, will be set in the loop
        position_index=-1, # Placeholder
        position_name="placeholder", # Placeholder
        batch_size=64, # Training batch size,
        weight_init_seed=42,
        # lr, betas, weight_init_range, weight_init_seed use defaults
    )

    # --- Create Config Object (Model details are now placeholders) ---
    config = TrainingConfig(
        data=DataConfig(
            probing_data_path=DATA_DIR / "probing_data.json",
            activations_dir=ACTS_DIR,
            meta_path=ACTS_DIR / "meta.json",
            output_dir=PROBE_OUTPUT_DIR # Specify where to save probes/results
        ),
        model=ModelConfig( # These values will be overwritten by meta.json
            name=MODEL_NAME_FOR_PATH,
            d_model=-1,  # Placeholder
            n_layers=-1    # Placeholder
        ),
        probe_config=default_probe_config,
        split_seed=42
        # Set wandb_project to None to disable logging
        # wandb_project=None,
        # wandb_entity="your_wandb_username_or_entity", # Set your entity if using wandb
    )

    # --- Run Training ---
    run_probe_training(config) 