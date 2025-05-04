import json
import random
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm.auto import tqdm
import wandb # Keep import here, usage is controlled by config
from scipy.stats import pearsonr

# --- Utility Functions ---

def setup_determinism(seed: int):
    """Sets seeds for reproducibility."""
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
    batch_size: int = 32
    weight_init_range: float = 0.02
    weight_init_seed: int = 42 # Seed for initializing probe weights

@dataclass
class TrainingConfig:
    """Overall configuration for the training run (single split)."""
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
    # Filter split_qids first to ensure they exist in the meta qid list
    valid_split_qids = [qid for qid in split_qids if qid in qid_to_row_index]
    if len(valid_split_qids) != len(split_qids):
         print(f"[Warning] {len(split_qids) - len(valid_split_qids)} QIDs from the target split were not found in meta.json for {activation_path}")

    row_indices = [qid_to_row_index[qid] for qid in valid_split_qids]

    if not row_indices:
        # Return an empty array with the correct dimensions if the split is empty
        return np.empty((0, d_model), dtype=memmap_dtype)

    # Load the memmap file
    try:
        # Ensure the shape matches exactly what was saved by get_acts.py
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
    # Need to handle potential empty row_indices if valid_split_qids was empty
    if not row_indices:
         return np.empty((0, d_model), dtype=memmap_dtype)

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
    # dtype: torch.dtype # Probe's internal dtype is float32
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
         # Ensure inputs to pearsonr are standard Python lists or 1D numpy arrays
         correlation, _ = pearsonr(np.array(all_labels), np.array(all_preds))
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
    # dtype: torch.dtype, # Activation dtype - not needed directly here
    train_loader: DataLoader,
    val_loader: DataLoader,
    # Pass training loop params (required)
    max_steps: int,
    patience: int,
    val_freq: int,
    # Optional args with defaults last
    test_loader: Optional[DataLoader] = None,
    use_wandb: bool = False, # Control wandb logging
    wandb_log_prefix: str = "", # Prefix for wandb keys (e.g., fold number)
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    """Trains a single linear probe and returns best state_dict and metrics."""

    run_name = f"L{probe_config.layer}_{probe_config.position_name}"
    # wandb is initialized outside this function now

    print(f"--- Training probe: {wandb_log_prefix} Layer {probe_config.layer}, Pos {probe_config.position_name} ---")

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

    # Need to handle potential empty train_loader
    if not train_loader.dataset:
         print("[Warning] Empty training dataset. Skipping training.")
         return None, {"error": "Empty training dataset"}

    train_iter = iter(train_loader)
    probe.train()

    pbar = tqdm(total=max_steps, desc=f"Train {wandb_log_prefix} L{probe_config.layer} P{probe_config.position_index}", leave=False)

    while steps_done < max_steps:
        try:
            batch = next(train_iter)
            batch_acts = batch.activations.to(device) # Ensure data on correct device
            batch_labels = batch.labels.to(device)
        except StopIteration:
            # Reinitialize iterator if dataset is not empty
             if len(train_loader.dataset) > 0:
                  train_iter = iter(train_loader)
                  try: # Try getting next batch immediately
                      batch = next(train_iter)
                      batch_acts = batch.activations.to(device)
                      batch_labels = batch.labels.to(device)
                  except StopIteration: # Dataset exhausted within one step? Very small dataset.
                      print("[Warning] Training dataset exhausted quickly. Stopping training.")
                      break
             else: # Should have been caught earlier, but defensive check
                 print("[Warning] Empty training dataset detected during loop. Stopping.")
                 break # Stop if dataset is empty


        optimizer.zero_grad()
        predictions = probe(batch_acts) # Forward pass uses float32
        loss = criterion(predictions, batch_labels)

        if math.isnan(loss.item()):
            print(f"[Warning] NaN loss detected at step {steps_done}. Skipping step.")
            # Don't increment step if skipping? Or stop? Let's skip step for now.
            steps_done += 1
            pbar.update(1)
            continue

        loss.backward()
        optimizer.step()
        steps_done += 1
        pbar.update(1)

        # Log training loss periodically
        if use_wandb and steps_done % val_freq == 0:
             log_data = {
                 f"{wandb_log_prefix}train_loss_step/L{probe_config.layer}_{probe_config.position_name}": loss.item(),
                 f"{wandb_log_prefix}step": steps_done # Step needs context if CV
             }
             wandb.log(log_data)

        # --- Validation ---
        # Need to handle potential empty val_loader
        if steps_done % val_freq == 0:
             if val_loader.dataset and len(val_loader.dataset) > 0:
                  val_loss, val_corr = evaluate_probe(probe, val_loader, criterion, device)
                  if math.isnan(val_loss):
                      print(f"[Warning] NaN validation loss at step {steps_done}. Stopping training for this probe.")
                      break # Stop training if validation loss is NaN

                  if use_wandb:
                      log_data = {
                          f"{wandb_log_prefix}val_loss/L{probe_config.layer}_{probe_config.position_name}": val_loss,
                          f"{wandb_log_prefix}val_pearson_r/L{probe_config.layer}_{probe_config.position_name}": val_corr,
                          f"{wandb_log_prefix}step": steps_done # Step needs context if CV
                      }
                      wandb.log(log_data)

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
             else:
                  # No validation data, can't do early stopping
                  # Option 1: Keep the latest model state
                  best_model_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
                  # Option 2: Raise error or warning
                  # print("[Warning] No validation data provided. Cannot perform early stopping. Saving latest model.")
                  # We'll proceed without early stopping if no val data

    pbar.close()
    if best_model_state is None and val_loader.dataset and len(val_loader.dataset) > 0 : # Only warn if val set existed but no improvement
         print("[Warning] No improvement found during training. Using final model state.")
         # Ensure final state is captured if loop finished normally without improvement
         best_model_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    elif best_model_state is None: # Handles case where val_loader was empty
        best_model_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}


    # --- Final Evaluation on Test Set ---
    final_metrics = {"best_val_loss": best_val_loss if (val_loader.dataset and len(val_loader.dataset) > 0) else None,
                     "steps_completed": steps_done}
    if test_loader is not None and test_loader.dataset and len(test_loader.dataset) > 0 and best_model_state is not None:
        print("  Evaluating on test set...")
        # Load best model state
        probe.load_state_dict(best_model_state)
        test_loss, test_corr = evaluate_probe(probe, test_loader, criterion, device)
        final_metrics["test_loss"] = float(test_loss) if not math.isnan(test_loss) else None
        final_metrics["test_pearson_r"] = float(test_corr) if not math.isnan(test_corr) else None
        if use_wandb:
             # Log final test metrics specific to this training run (e.g., fold)
             wandb.log({
                 f"{wandb_log_prefix}final_test_loss/L{probe_config.layer}_{probe_config.position_name}": final_metrics["test_loss"],
                 f"{wandb_log_prefix}final_test_pearson_r/L{probe_config.layer}_{probe_config.position_name}": final_metrics["test_pearson_r"]
             })
        print(f"  Test Loss: {test_loss:.4f}, Test Pearson R: {test_corr:.4f}")
    else:
         print("  Skipping test set evaluation (no test data or no valid model state).")
         final_metrics["test_loss"] = None
         final_metrics["test_pearson_r"] = None


    print(f"--- Finished training: {wandb_log_prefix} Layer {probe_config.layer}, Pos {probe_config.position_name} ---")
    print(f"Final Metrics: {final_metrics}")

    return best_model_state, final_metrics 