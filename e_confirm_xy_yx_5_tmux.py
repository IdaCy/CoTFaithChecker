#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ───────────────────────────────────────────────
# 0. Manual configuration
# ───────────────────────────────────────────────
from pathlib import Path
import torch

DATA_ROOT = Path("data/chainscope/questions_json")
TEMPLATE_PATH = Path("data/chainscope/templates/instructions.json")
LOG_DIR = Path("logs")
OUT_DIR = Path("e_confirm_xy_yx/outputs")          # completions, verification, matches
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# choose folder subsets
MAIN_TYPE = "gt"
MAIN_TYPE_2 = "lt"
DATASETS = [MAIN_TYPE + "_NO_1", MAIN_TYPE + "_YES_1", MAIN_TYPE_2 + "_NO_1", MAIN_TYPE_2 + "_YES_1"]

BATCH_SIZE = 64
MAX_NEW_TOKENS = None

# ─── multi-run & sampling ────────────────────────────────────────
N_RUNS      = 10      # generate 10 reasoning chains per question
TEMPERATURE = 0.7     # sampling temperature
TOP_P       = 0.9     # nucleus-sampling top-p
# ──────────────────────────────────────────────────────────────────────

OUT_GEN = Path("e_confirm_xy_yx/outputs/" + MAIN_TYPE + "_" + MAIN_TYPE_2 + "_completions_" + str(N_RUNS))

SAVE_HIDDEN, SAVE_ATTN = False, False
HIDDEN_LAYERS, ATTN_LAYERS = [0, -1], [0, -1]   # ignored unless above switches True
N_VERIFY = 0   # 0 == verify all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────
# 1. Load model & tokenizer  (your helper)
# ───────────────────────────────────────────────
from a_confirm_posthoc.utils.model_handler import load_model_and_tokenizer

model, tokenizer, model_name, device = load_model_and_tokenizer(MODEL_PATH)
model.to(device)


# In[2]:


from e_confirm_xy_yx.main.data_loader import get_dataset_files

# 0. Extra toggle
CLUSTERS = ["arts"]   # no "no_wm"

# 2. Collect dataset files
dataset_files = get_dataset_files(
    DATA_ROOT,
    DATASETS,
    clusters=CLUSTERS,          # ← NEW ARG
)

# 5. Verify – point to aggregated cluster outputs
completion_files = sorted(
    (OUT_DIR / "completions" / "clusters").glob("*_completions.json")
)

# 6. Match YES vs NO on cluster files
verified_files = sorted((OUT_DIR / "verified").glob("*_verified.json"))

pairs = [
    (vf, vf.parent / vf.name.replace("_NO_", "_YES_"))
    for vf in verified_files
    if "_NO_" in vf.name
    and (vf.parent / vf.name.replace("_NO_", "_YES_")).exists()
]


# In[3]:


# ───────────────────────────────────────────────
# 2. Collect dataset files
# ───────────────────────────────────────────────
#from e_confirm_xy_yx.main.data_loader import get_dataset_files
#dataset_files = get_dataset_files(DATA_ROOT, DATASETS)

# ───────────────────────────────────────────────
# 3. Prepare prompt builder
# ───────────────────────────────────────────────
from e_confirm_xy_yx.main.prompt_builder import PromptBuilder
pb = PromptBuilder(template_path=TEMPLATE_PATH, style="instr-v0", mode="cot")

# ───────────────────────────────────────────────
# 4. Run inference
# ───────────────────────────────────────────────
from e_confirm_xy_yx.main.inference import run_inference

run_inference(
    dataset_files=dataset_files,
    prompt_builder=pb,
    model=model,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
    save_hidden=SAVE_HIDDEN,
    hidden_layers=HIDDEN_LAYERS,
    save_attention=SAVE_ATTN,
    attn_layers=ATTN_LAYERS,
    output_dir=OUT_GEN,
    n_runs=N_RUNS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
)


# In[ ]:


"""
# ───────────────────────────────────────────────
# 5. Verify model answers
# ───────────────────────────────────────────────
from e_confirm_xy_yx.main.verifier import run_verification
completion_files = sorted((OUT_DIR / "completions").glob("*_completions.json"))

run_verification(
    completion_files=completion_files,
    n_questions=N_VERIFY,
    output_dir=OUT_DIR / "verified",
)
"""


# In[ ]:


"""
# ───────────────────────────────────────────────
# 6. Cross-match YES vs NO answers
# ───────────────────────────────────────────────
from e_confirm_xy_yx.main.match_checker import check_matches
verified_files = sorted((OUT_DIR / "verified").glob("*_verified.json"))

# pair them: every gt_NO_X file with its matching gt_YES_X (adapt if lt)
pairs = [
    (
        vf,
        vf.parent
        / vf.name.replace("gt_NO", "gt_YES")
    )
    for vf in verified_files
    if "_NO_" in vf.name
]

for no_file, yes_file in pairs:
    out_match = (
        OUT_DIR
        / "matches"
        / f"{no_file.stem.replace('_verified','')}_match.json"
    )
    out_match.parent.mkdir(parents=True, exist_ok=True)
    check_matches(no_file, yes_file, out_match)
"""

