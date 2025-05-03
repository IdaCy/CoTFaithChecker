"""
!!! FIRST CALL THIS TO SET UP ACCELERATE !!! ->
accelerate config
<- select: this machine, multi-gpu, 1 node, everything default, 4 GPUs (?), bf16

then run:
nohup accelerate launch a_confirm_posthoc/parallelization/driver1007.py \
     > logs/5_f1_2k_xyyx_$(date +%Y%m%d_%H%M%S).log 2>&1 &

no accelerate:
nohup python a_confirm_posthoc/parallelization/driver1008.py \
     > logs/5_f1_2k_xyyx_$(date +%Y%m%d_%H%M%S).log 2>&1 &

i mean, i'm a fan of nohup, you can also do tmux or have it run in the foreground tho
"""

import sys, pathlib, os, logging
from pathlib import Path

import logging
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
import socket, os, sys
from accelerate import Accelerator

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print("Working dir:", PROJECT_ROOT)

from a_confirm_posthoc.parallelization.pipeline import generate_dataset_completions
from a_confirm_posthoc.parallelization.model_handler import load_model_and_tokenizer
from accelerate.utils import gather_object
from a_confirm_posthoc.eval.llm_verificator import run_verification
from a_confirm_posthoc.eval.switch_check import run_switch_check
from a_confirm_posthoc.eval.llm_hint_verificator import run_hint_verification

LOG_FILE = "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(process)d - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w")]
)

accelerator = Accelerator()
if accelerator.is_main_process:
    logging.getLogger().addHandler(logging.StreamHandler())

print(f"on host {socket.gethostname()} (PID {os.getpid()}) ===",)
print("starting at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

#model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
#model, tokenizer, model_name, device = load_model_and_tokenizer(model_path)
model, tokenizer, model_name, _ = load_model_and_tokenizer(model_path)
model, tokenizer = accelerator.prepare(model, tokenizer)
device = accelerator.device

dataset_name = "mmlu"
#hint_types = ["none", "sycophancy", "unethical_information", "induced_urgency"]
#hint_types = ["sycophancy", "unethical_information", "induced_urgency"]
hint_types = ["none"]
n_questions = 1008

"""print("generating completions at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

generate_dataset_completions(
    accelerator=accelerator,
    model=model,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device,
    dataset_name=dataset_name,
    hint_types=hint_types,
    batch_size=32,          # per-GPU !
    max_new_tokens=2048,
    n_questions=n_questions
)
print("verifying completions at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
"""

# Run llm verification to get the final model answers
# Note that this will drop the results that are N/A (eg the model never stopped reasoning)
run_verification(dataset_name, hint_types, model_name, n_questions)

hint_types = ["none", "sycophancy"]
print("running switch check at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

# Check if the model switches between none and the other hint types
# [1:] because we don't want to check the none hint type as it's the baseline
run_switch_check(dataset_name, hint_types[1:], model_name, n_questions)

print("running hint verification at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
# Verify if the model verbalizes the hint
# [1:] because we don't want to check the none hint type as it's the baseline
run_hint_verification(dataset_name, hint_types[1:], model_name, n_questions)

print("done at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))
