#!/usr/bin/env python
# coding: utf-8


import sys, pathlib, os, logging, json
# ── make the project root importable ─────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))          # <── key line
os.chdir(PROJECT_ROOT)
print("Working dir:", PROJECT_ROOT)

# ───────────────────────────────────────────────
# 0. Manual configuration
# ───────────────────────────────────────────────
print("CoTFaithChecker/run_xyyx_several.py starting!")
from pathlib import Path
import torch
from datetime import datetime
import random, numpy as np

"""
starting with:
nohup python -u run_xyyx.py \
      > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

kill:
pgrep -f run_xyyx_several.py
-> kill

tail -F logs/run_*.log logs/*inference_*.log
watch -n 5 nvidia-smi
"""
print(f"[{datetime.now().isoformat()}] starting setup", flush=True)

DATA_ROOT = Path("data/chainscope/questions_json")
TEMPLATE_PATH = Path("data/chainscope/templates/instructions.json")
LOG_DIR = Path("logs")
OUT_DIR = Path("h_hidden_states/outputs/xyyx_deterministic")          # completions, verification, matches
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# choose folder subsets
MAIN_TYPE = "gt"
MAIN_TYPE_2 = "lt"
DATASETS = [MAIN_TYPE + "_NO_1", MAIN_TYPE + "_YES_1", MAIN_TYPE_2 + "_NO_1", MAIN_TYPE_2 + "_YES_1"]
#DATASETS = [MAIN_TYPE + "_NO_1"]

BATCH_SIZE = 32
MAX_NEW_TOKENS = None

# ─── multi-run & sampling ────────────────────────────────────────
N_RUNS      = 1      # generate 10 reasoning chains per question
TEMPERATURE = 1.0     # sampling temperature
TOP_P       = 1.0     # nucleus-sampling top-p
SEED        = 0
# reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# ──────────────────────────────────────────────────────────────────────

#OUT_GEN = Path(OUT_DIR / MAIN_TYPE + "_" + MAIN_TYPE_2 + "_completions_" + str(N_RUNS))
OUT_GEN = OUT_DIR / f"{MAIN_TYPE}_{MAIN_TYPE_2}_completions_{N_RUNS}"

SAVE_HIDDEN, SAVE_ATTN = True, False
HIDDEN_LAYERS, ATTN_LAYERS = list(range(4, 33, 4)), []
N_VERIFY = 0   # 0 == verify all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────
# 1. Load model & tokenizer  (your helper)
# ───────────────────────────────────────────────
print(f"[{datetime.now().isoformat()}] loading model & tokenizer", flush=True)
from h_hidden_states.hint.model_handler import load_model_and_tokenizer

model, tokenizer, model_name, device = load_model_and_tokenizer(MODEL_PATH)
model.to(device)


from e_confirm_xy_yx.main.data_loader import get_dataset_files

# 0. Extra toggle
CLUSTERS = ["arts", "ny", "human"]   # no "no_wm"

# ───────────────────────────────────────────────
# 2. Collect dataset files
# ───────────────────────────────────────────────
print(f"[{datetime.now().isoformat()}] loading data", flush=True)

#from e_confirm_xy_yx.main.data_loader import get_dataset_files
#dataset_files = get_dataset_files(DATA_ROOT, DATASETS)
dataset_files = get_dataset_files(
    DATA_ROOT,
    DATASETS,
    clusters=CLUSTERS,
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

# ───────────────────────────────────────────────
# 3. Prepare prompt builder
# ───────────────────────────────────────────────
from e_confirm_xy_yx.main.prompt_builder import PromptBuilder
pb = PromptBuilder(template_path=TEMPLATE_PATH, style="instr-v0", mode="cot")

# ───────────────────────────────────────────────
# 4. Run inference
# ───────────────────────────────────────────────
print(f"[{datetime.now().isoformat()}] running inference", flush=True)

from h_hidden_states.xyyx.inference import run_inference

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
    n_samples=N_RUNS,
)
print(f"[{datetime.now().isoformat()}] done!", flush=True)


##### New long-generation (2048) multi run NON-DETERMINISTIC

print("CoTFaithChecker/run_xyyx_several.py ** 2 ** starting!")
print(f"[{datetime.now().isoformat()}] starting setup", flush=True)

DATA_ROOT = Path("data/chainscope/questions_json")
TEMPLATE_PATH = Path("data/chainscope/templates/instructions.json")
LOG_DIR = Path("logs")
OUT_DIR = Path("h_hidden_states/outputs/xyyx_nondeterministic")
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# choose folder subsets
MAIN_TYPE = "gt"
MAIN_TYPE_2 = "lt"
DATASETS = [MAIN_TYPE + "_NO_1", MAIN_TYPE + "_YES_1", MAIN_TYPE_2 + "_NO_1", MAIN_TYPE_2 + "_YES_1"]
#DATASETS = [MAIN_TYPE + "_NO_1"]

BATCH_SIZE = 32
MAX_NEW_TOKENS = None

# ─── multi-run & sampling ────────────────────────────────────────
N_RUNS      = 10      # generate 10 reasoning chains per question
TEMPERATURE = 0.7     # sampling temperature
TOP_P       = 1.0     # nucleus-sampling top-p
SEED        = 0

# reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# ──────────────────────────────────────────────────────────────────────

#OUT_GEN = Path(OUT_DIR / MAIN_TYPE + "_" + MAIN_TYPE_2 + "_completions_" + str(N_RUNS))
OUT_GEN = OUT_DIR / f"{MAIN_TYPE}_{MAIN_TYPE_2}_completions_{N_RUNS}"

SAVE_HIDDEN, SAVE_ATTN = True, False
HIDDEN_LAYERS, ATTN_LAYERS = list(range(4, 33, 4)), []
N_VERIFY = 0   # 0 == verify all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────
# 1. Load model & tokenizer  (your helper)
# ───────────────────────────────────────────────
print(f"[{datetime.now().isoformat()}] loading model & tokenizer", flush=True)

# 0. Extra toggle
CLUSTERS = ["arts", "ny"]   # no "no_wm"

# ───────────────────────────────────────────────
# 2. Collect dataset files
# ───────────────────────────────────────────────
print(f"[{datetime.now().isoformat()}] loading data", flush=True)

#from e_confirm_xy_yx.main.data_loader import get_dataset_files
#dataset_files = get_dataset_files(DATA_ROOT, DATASETS)
dataset_files = get_dataset_files(
    DATA_ROOT,
    DATASETS,
    clusters=CLUSTERS,
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

# ───────────────────────────────────────────────
# 3. Prepare prompt builder
# ───────────────────────────────────────────────
from e_confirm_xy_yx.main.prompt_builder import PromptBuilder
pb = PromptBuilder(template_path=TEMPLATE_PATH, style="instr-v0", mode="cot")

# ───────────────────────────────────────────────
# 4. Run inference
# ───────────────────────────────────────────────
print(f"[{datetime.now().isoformat()}] running inference", flush=True)

from h_hidden_states.xyyx.inference import run_inference

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
    n_samples=N_RUNS,
)
print(f"[{datetime.now().isoformat()}] done!", flush=True)





