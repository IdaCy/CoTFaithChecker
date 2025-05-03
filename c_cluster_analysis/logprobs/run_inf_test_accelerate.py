
"""
nohup accelerate launch c_cluster_analysis/logprobs/run_inf_test_accelerate.py \
     > logs/generate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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

from accelerate.utils import gather_object

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



from c_cluster_analysis.logprobs.sentence_level_inference import (
    load_model_and_tokenizer, run_batch_from_files
)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model, tok, _, _ = load_model_and_tokenizer(MODEL)

questions_file = "data/mmlu/input_mcq_data.json"
hints_file     = "data/mmlu/hints_sycophancy.json" # can be None!
whitelist_file = "/root/CoTFaithChecker/c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B/unverb_ids_2001_sycophancy.json"

out_none_file  = "c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B/attn_sentence_level_results_none_unverbalized_ids.json"

hints_file     = None

print("Running inference with no hints")
run_batch_from_files(
    model, tok,
    questions_file=questions_file,
    hints_file=hints_file,
    output_file=out_none_file,
    whitelist_file=whitelist_file,
    max_questions=None,
    max_new_tokens=2048,
)
print("Done", out_none_file)

