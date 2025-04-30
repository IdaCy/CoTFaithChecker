
#%%
# Setup and Imports
import nbformat
import json
import torch as t
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.auto import tqdm
from scipy.stats import mannwhitneyu, ttest_ind
from transformers import AutoTokenizer  # Use standard transformers tokenizer for DeepSeek
from transformer_lens import HookedTransformer  # Use standard TL for DeepSeek
from sae_lens import SAE, ActivationsStore
from huggingface_hub import snapshot_download
from rich import print as rprint
from rich.table import Table

#%%
# --- Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_PATH_ORIGINAL = "meta-llama/Llama-3.1-8B-Instruct"  # Non-reasoning model
SAE_RELEASE = "llama_scope_r1_distill"
# Found this specific SAE ID by browsing the HF repo structure for llama_scope_r1_distill
# Adjust if this is not the exact one you intended.
SAE_ID = "l15r_400m_slimpajama_400m_openr1_math"
DATA_FILE = "fff_saes/data/input/mmlu/DeepSeek-R1-Distill-Llama-8B/sycophancy/1001.json"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
STAT_TEST_ALPHA = 0.05 # Significance level for statistical tests
DEMO_MODE = True

print(f"Using device: {DEVICE}")

# Disable gradients for inference
t.set_grad_enabled(False)

#%%
# Load Model and SAE

print(f"Loading model: {MODEL_NAME}...")
# Note: HookedSAETransformer might not directly support DeepSeek.
# Using standard HookedTransformer and checking compatibility.
# If issues arise, manual hooking might be needed, but let's try standard first.

# model_path = snapshot_download(
#     repo_id=MODEL_NAME,
#     local_dir=MODEL_PATH_ORIGINAL,
#     local_dir_use_symlinks=False
# )

try:
    # Deepseek requires trust_remote_code
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH_ORIGINAL,
        local_files_only=True,  # Set to True if using local models
        dtype=t.bfloat16,
        default_padding_side='left',
        device=DEVICE
    )
except Exception as e:
    print(f"Error loading model with HookedTransformer: {e}")
    print("Attempting loading with AutoModelForCausalLM and wrapping later if needed.")
    # Fallback or specific handling might be required if HookedTransformer fails.
    # For now, proceed assuming it works or adjust based on the error.
    raise

#%%
%cd ..


#%%
print(f"Loading SAE: {SAE_RELEASE} / {SAE_ID}...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE,
)

# We need the exact hook point name from the SAE config
hook_point = sae.cfg.hook_name
hook_layer = sae.cfg.hook_layer
print(f"SAE Hook Point: {hook_point}")
print(f"SAE Hook Layer: {hook_layer}")

# Ensure model tokenizer matches expected (or handle differences)
# The SAE was likely trained with a specific tokenizer associated with the base model.
# Let's load the tokenizer separately to be sure.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Common practice for models without a specific pad token


#%%
# Load and Prepare Data

print(f"Loading data from: {DATA_FILE}")
with open(DATA_FILE, 'r') as f:
    data = json.load(f)
    # only keep the first 5 items
    if DEMO_MODE:
        data = data[:5]

# Assuming the JSON structure is a list of dicts like:
# [{'faithful_completion': '...', 'unfaithful_completion': '...'}, ...]
# Or potentially includes the prompt/question as well. Adjust parsing as needed.

faithful_texts = [item.get('unhinted_completion', '') for item in data]
unfaithful_texts = [item.get('hinted_completion', '') for item in data]

# Filter out any empty strings if they exist
faithful_texts = [text for text in faithful_texts if text]
unfaithful_texts = [text for text in unfaithful_texts if text]

print(f"Found {len(faithful_texts)} faithful completions.")
print(f"Found {len(unfaithful_texts)} unfaithful completions.")

# Basic check: Ensure we have pairs to compare
if len(faithful_texts) != len(unfaithful_texts) or len(faithful_texts) == 0:
    print("Warning: Number of faithful and unfaithful texts differ or is zero. Check data format.")
    # Decide how to proceed: error out, pair up based on index, etc.
    # For now, let's proceed but analysis might be flawed if lists aren't paired correctly.

#%%
data

#%%
# Helper function to get activations for a list of texts

def get_activations(texts: list[str], batch_size=8):
    """
    Gets SAE activations for a list of texts using ActivationsStore.
    Returns a dictionary mapping latent_idx to a list of its positive activations.
    """
    # Configure ActivationsStore
    # Need enough storage tokens to cover the texts, adjust if needed
    n_batches_in_buffer = 4 # Controls memory usage, lower if OOM
    # store_batch_size = batch_size * model.cfg.n_ctx # Tokens per batch in store # Not needed for from_sae

    # Estimate total tokens needed - rough estimate, might need adjustment
    # avg_tokens_per_text = np.mean([len(tokenizer.encode(t)) for t in texts]) if texts else 0
    # total_tokens_needed = int(avg_tokens_per_text * len(texts))
    # if total_tokens_needed == 0: return {} # Handle empty texts

    print("Setting up ActivationsStore...")
    # Use the from_sae constructor
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae, # Pass the loaded SAE object
        streaming=True, # Assuming streaming is desired, adjust if data is local/small
        store_batch_size_prompts=batch_size, # How many prompts to process with model at once
        n_batches_in_buffer=n_batches_in_buffer,
        device=DEVICE,
    )


    # Create a generator function for the ActivationsStore - REMOVED
    # def text_token_generator():
    #     for i in range(0, len(texts), batch_size):
    #         batch_texts = texts[i:i+batch_size]
    #         # Tokenize manually for the generator
    #         tokenized_batch = tokenizer(
    #             batch_texts,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #             max_length=activation_store.context_size
    #         ).input_ids
    #         yield tokenized_batch

    print(f"Getting activations for {len(texts)} texts...")
    # Dictionary to store all positive activations for each latent
    # Format: {latent_idx: [act1, act2, ...]}
    latent_activations = {i: [] for i in range(sae.cfg.d_sae)}
    total_processed_tokens = 0

    # Use the generator with the store - REVISED LOOP
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        if not batch_texts: continue # Skip empty batches

        try:
            # Tokenize the current batch
            batch_tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=activation_store.context_size # Use context size from store/SAE
            ).input_ids.to(DEVICE)

            # Get a batch of SAE activations using get_activations
            acts = activation_store.get_activations(batch_tokens) # Note: using get_sae_activations which returns shape (batch, seq, d_sae)
            # Shape: (batch_size, seq_len, d_sae)
            acts = acts.to(DEVICE) # Ensure it's on the right device

            # Iterate through latents and collect positive activations
            positive_acts = acts[acts > 0]
            positive_indices = (acts > 0).nonzero(as_tuple=False) # Get indices (batch, seq, latent), use as_tuple=False

            for idx_tuple, value in zip(positive_indices.tolist(), positive_acts.tolist()):
                # idx_tuple structure: [batch_in_this_acts_tensor, seq_pos, latent_idx]
                latent_idx = idx_tuple[2]
                if latent_idx in latent_activations: # Check just in case
                     latent_activations[latent_idx].append(value)

            total_processed_tokens += acts.shape[0] * acts.shape[1]

        except StopIteration: # Should not happen with this loop structure
            print("Text generator exhausted.")
            break
        except Exception as e:
            print(f"Error during activation retrieval in batch {i}: {e}")
            # Decide how to handle errors, e.g., skip batch, stop processing
            continue # Simple handling: skip batch on error

    print(f"Finished processing. Total tokens analyzed (approx): {total_processed_tokens}")
    # Clean up store explicitly if needed (helps with memory)
    del activation_store
    if DEVICE == 'cuda':
        t.cuda.empty_cache()

    # Filter out latents that never activated
    active_latents = {k: v for k, v in latent_activations.items() if v}
    print(f"Found {len(active_latents)} latents with positive activations.")
    return active_latents

#%%
# Get Activations for Faithful and Unfaithful Sets

print("--- Getting Activations for Faithful Set ---")
faithful_activations = get_activations(faithful_texts)

print("\n--- Getting Activations for Unfaithful Set ---")
unfaithful_activations = get_activations(unfaithful_texts)

#%%
# Perform Statistical Comparison

print("\n--- Performing Statistical Comparison ---")

results = []
common_latents = set(faithful_activations.keys()) & set(unfaithful_activations.keys())
unique_faithful = set(faithful_activations.keys()) - set(unfaithful_activations.keys())
unique_unfaithful = set(unfaithful_activations.keys()) - set(faithful_activations.keys())

print(f"Commonly active latents: {len(common_latents)}")
print(f"Unique to faithful: {len(unique_faithful)}")
print(f"Unique to unfaithful: {len(unique_unfaithful)}")

for latent_idx in tqdm(common_latents, desc="Comparing common latents"):
    acts_faithful = np.array(faithful_activations[latent_idx])
    acts_unfaithful = np.array(unfaithful_activations[latent_idx])

    # Skip if not enough data points for a meaningful comparison
    if len(acts_faithful) < 5 or len(acts_unfaithful) < 5:
        continue

    # Calculate basic stats
    mean_f = np.mean(acts_faithful)
    mean_u = np.mean(acts_unfaithful)
    std_f = np.std(acts_faithful)
    std_u = np.std(acts_unfaithful)
    count_f = len(acts_faithful)
    count_u = len(acts_unfaithful)

    # Perform Mann-Whitney U test (non-parametric)
    try:
        stat, p_value = mannwhitneyu(acts_faithful, acts_unfaithful, alternative='two-sided')
    except ValueError as e:
        # Can happen if all values are identical in one sample
        print(f"Skipping latent {latent_idx} due to Mann-Whitney U error: {e}")
        stat, p_value = np.nan, np.nan


    # Optional: t-test (assumes normality, might be less robust)
    # stat_t, p_value_t = ttest_ind(acts_faithful, acts_unfaithful, equal_var=False) # Welch's t-test

    results.append({
        'latent_idx': latent_idx,
        'mean_faithful': mean_f,
        'mean_unfaithful': mean_u,
        'std_faithful': std_f,
        'std_unfaithful': std_u,
        'count_faithful': count_f,
        'count_unfaithful': count_u,
        'mannwhitney_stat': stat,
        'p_value': p_value,
        'significant': p_value < STAT_TEST_ALPHA if not np.isnan(p_value) else False
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p_value').reset_index(drop=True)

print("\n--- Comparison Results ---")
significant_results = results_df[results_df['significant']]
print(f"Found {len(significant_results)} latents with significantly different activation distributions (p < {STAT_TEST_ALPHA}).")

# Display top N significant results
n_top_results = 20
print(f"\nTop {n_top_results} most significant differentiating latents:")
print(significant_results.head(n_top_results).to_markdown(index=False))


#%%
# Visualize Distributions for Top Latents (Optional)

print("\n--- Visualizing Top Differentiating Latent Distributions ---")
n_plots = 5 # Number of top latents to plot

top_significant_latents = significant_results.head(n_plots)['latent_idx'].tolist()

plot_data = []
for latent_idx in top_significant_latents:
    if latent_idx in faithful_activations:
        for act in faithful_activations[latent_idx]:
            plot_data.append({'latent': f"Latent {latent_idx}", 'activation': act, 'condition': 'Faithful'})
    if latent_idx in unfaithful_activations:
        for act in unfaithful_activations[latent_idx]:
            plot_data.append({'latent': f"Latent {latent_idx}", 'activation': act, 'condition': 'Unfaithful'})

if plot_data:
    plot_df = pd.DataFrame(plot_data)
    fig = px.histogram(plot_df, x='activation', color='condition',
                       facet_col='latent', barmode='overlay',
                       title=f'Activation Distributions for Top {n_plots} Differentiating Latents',
                       histnorm='probability density', nbins=50)
    fig.update_layout(yaxis_title="Density")
    fig.update_traces(opacity=0.7)
    fig.show()
else:
    print("No data available for plotting top significant latents.")

print("\n--- Analysis Complete ---")

# %%
# plot_df = pd.DataFrame(plot_data)
fig = px.histogram(plot_df, x='activation', color='condition',
                    facet_col='latent', barmode='overlay',
                    title=f'Activation Distributions for Top {n_plots} Differentiating Latents',
                    histnorm='probability density', nbins=50)
fig.update_layout(yaxis_title="Density")
fig.update_traces(opacity=0.7)
fig.show()

# %%
