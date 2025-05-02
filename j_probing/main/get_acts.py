"""Cache hidden activations for selected token positions across all layers.

This script loads the probing dataset produced by `create_probing_dataset.py`,
feeds every prompt through the chosen model using **TransformerLens** and
stores (layer, hidden_size) activations at three token positions for every
prompt:
    0 → assistant token (last token in the prompt)
    1 → think token      (second–to–last token)
    2 → hint token       (first token that includes substring 'hint')

The result is one tensor per layer written to `<output_dir>/layer_{L:02d}.pt`
with shape `(N_prompts, hidden_size, 3)` – the last dim corresponds to the
three key positions in the above order.
A `meta.json` file lists the question_ids in the same row-order.

NOTE: Argparse is deliberately avoided.  At the bottom of the file we call
`main()` with placeholder paths/params – adjust them before executing in your
own environment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm  # type: ignore

# --------------------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------------------

def load_probing_data(filepath: str | Path) -> List[Dict[str, Any]]:
    """Return list of dicts as stored in probing_data.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_batches(lst: List[Any], batch_size: int):
    """Simple generator that yields consecutive slices of `lst`."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


# --------------------------------------------------------------------------------------
# Activation extraction
# --------------------------------------------------------------------------------------

def extract_batch_activations(
    model: HookedTransformer,
    prompts: List[str],
    token_positions: List[List[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Run a batch through the model and collect activations.

    Returns:
        List[length = n_layers] of tensors with shape (batch, hidden_size, 3)
    """
    # Tokenise
    tokens = model.to_tokens(prompts, prepend_bos=False).to(device)

    # Run forward pass with cache
    with torch.no_grad():
        _logits, cache = model.run_with_cache(
            tokens,
            return_type=None,
            device=device,
            remove_batch_dim=False,
        )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Prepare per-layer containers
    acts_per_layer = [torch.empty((len(prompts), d_model, 3), dtype=dtype, device="cpu") for _ in range(n_layers)]

    # Iterate over batch items to slice positions
    for b_idx, (tok_idxs, pos_list) in enumerate(zip(tokens, token_positions)):
        seq_len = tok_idxs.size(0)
        # Unpack expected positions; clamp to seq_len-1 if out-of-range
        a_idx, t_idx, h_idx = pos_list
        a_idx = min(max(a_idx, 0), seq_len - 1)
        t_idx = min(max(t_idx, 0), seq_len - 1)
        h_idx = min(max(h_idx, 0), seq_len - 1)

        for layer in range(n_layers):
            resid: torch.Tensor = cache[f"resid_post", layer][b_idx]  # (seq_len, d_model)
            # Grab activations at our three positions & move to CPU right away
            acts_per_layer[layer][b_idx, :, 0] = resid[a_idx].to("cpu")
            acts_per_layer[layer][b_idx, :, 1] = resid[t_idx].to("cpu")
            acts_per_layer[layer][b_idx, :, 2] = resid[h_idx].to("cpu")

    # Explicitly free GPU cache of this forward pass
    del cache, tokens
    torch.cuda.empty_cache()

    return acts_per_layer


# --------------------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------------------

def run(
    probing_json_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    batch_size: int = 4,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load prompts + positions
    data = load_probing_data(probing_json_path)
    prompts = [d["prompt"] for d in data]
    token_pos = [d["token_pos"] for d in data]
    qids = [d["question_id"] for d in data]

    # Persist question ordering
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"question_ids": qids}, f, indent=2)

    # 2) Load model via TransformerLens
    print(f"Loading model {model_name} …")
    model = HookedTransformer.from_pretrained(model_name, device=str(device), dtype=dtype)
    model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_prompts = len(prompts)

    # 3) Pre-allocate mem-maps (one file per layer)
    layer_files = []
    for layer in range(n_layers):
        fpath = output_dir / f"layer_{layer:02d}.bin"
        # Create mem-map so we can write incrementally
        mm = torch.memmap(
            filename=str(fpath),
            mode="w+",
            dtype=dtype,
            shape=(n_prompts, d_model, 3),
        )
        layer_files.append(mm)
    print(f"Allocated mem-maps for {n_layers} layers in {output_dir.relative_to(Path.cwd())}")

    # 4) Iterate over batches and fill mem-maps
    for start in tqdm(range(0, n_prompts, batch_size), desc="Batches"):
        end = min(start + batch_size, n_prompts)
        batch_prompts = prompts[start:end]
        batch_pos = token_pos[start:end]

        acts_batch = extract_batch_activations(
            model, batch_prompts, batch_pos, device=torch.device(device), dtype=dtype
        )

        for layer, acts in enumerate(acts_batch):
            layer_files[layer][start:end] = acts  # type: ignore[index]

    # 5) Flush & clean up
    for mm in layer_files:
        mm.flush()
    print("Done – activations cached.")


# --------------------------------------------------------------------------------------
# Example invocation – edit the paths/params below before running!
# --------------------------------------------------------------------------------------

def main():
    PROBING_JSON = (
        "j_probing/data/mmlu/DeepSeek-R1-Distill-Llama-8B/sycophancy/301/probing_data.json"
    )
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    OUTPUT_DIR = "j_probing/acts/mmlu/DeepSeek-R1-Distill-Llama-8B/sycophancy/301"

    run(
        probing_json_path=PROBING_JSON,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        batch_size=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )


if __name__ == "__main__":
    main()
