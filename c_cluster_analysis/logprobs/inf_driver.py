from datetime import datetime, timezone
import socket, os, sys
from pathlib import Path

"""
nohup python -u c_cluster_analysis/logprobs/inf_driver.py \
    > logs/inf_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

start_ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
print(
    f"\nSegmentation-annotate job started {start_ts} "
    f"on host {socket.gethostname()} (PID {os.getpid()}) ===",
    flush=True,
)
from datetime import datetime
from zoneinfo import ZoneInfo

ts_london  = datetime.now(ZoneInfo("Europe/London"))
ts_florida = datetime.now(ZoneInfo("America/New_York"))
print(ts_london.isoformat(timespec="seconds"))
print(ts_florida.isoformat(timespec="seconds"))

print("Using os.getcwd():", os.getcwd())
print("Using Path.cwd():",  Path.cwd())

print("\nDirectory contents:")
for p in Path.cwd().iterdir():
    print("  ", p.name)

# move two directories up from the location of this file
project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)
print("Working directory set to:", os.getcwd())
from pathlib import Path
import sys, os

project_root = Path(__file__).resolve().parents[2]   # …/CoTFaithChecker
os.chdir(project_root)                               # good for file paths

# 👇 Make the repo root visible to the import machinery
sys.path.insert(0, str(project_root))

# (optional) keep `src` if you really need it
# sys.path.append(str(project_root / "src"))

from c_cluster_analysis.logprobs.sentence_level_inference import (
    load_model_and_tokenizer, run_batch_from_files
)


import sys, os, json
sys.path.append(os.path.abspath("src"))


input_count = "2001"
verbnonverb = "unverbalized"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model, tok, _, _ = load_model_and_tokenizer(MODEL)

data_dir = "data/mmlu"
sycophancy_dir = data_dir + "/DeepSeek-R1-Distill-Llama-8B/sycophancy"
output_dir = "c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B"

questions_file = data_dir + "/input_mcq_data.json"
hints_file     = data_dir + "/hints_sycophancy.json" # can be None!
whitelist_file = output_dir + "/filter/"+verbnonverb+"_ids_"+input_count+".json"


#out_file       = "c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B/attn/sentence_level_results_sycophancy_unverbalized_ids.json"

print("Running sycophancy")
hint_type = "sycophancy"
run_batch_from_files(
    model, tok,
    questions_file=questions_file,
    hints_file=hints_file,
    output_file=output_dir+"/"+hint_type+"_"+verbnonverb+"_"+input_count+".json",
    whitelist_file=whitelist_file,
    max_questions=None,
    max_new_tokens=2048,
)

print("Done", out_file)

print("Running none")
hint_type = "none"
run_batch_from_files(
    model, tok,
    questions_file=questions_file,
    hints_file=hints_file,
    output_file=output_dir+"/"+hint_type+"_"+verbnonverb+"_"+input_count+".json",
    whitelist_file=whitelist_file,
    max_questions=None,
    max_new_tokens=2048,
)

print("Done", out_file)


###################################SECOND#############################################

print("Running sycophancy")
input_count = "3001"
questions_file = data_dir + "/input_mcq_data_3000.json"

hint_type = "sycophancy"
run_batch_from_files(
    model, tok,
    questions_file=questions_file,
    hints_file=hints_file,
    output_file=output_dir+"/"+hint_type+"_"+verbnonverb+"_"+input_count+".json",
    whitelist_file=whitelist_file,
    max_questions=None,
    max_new_tokens=2048,
)

print("Done", out_file)

print("Running none")
hints_file = None
hint_type = "none"
run_batch_from_files(
    model, tok,
    questions_file=questions_file,
    hints_file=hints_file,
    output_file=output_dir+"/"+hint_type+"_"+verbnonverb+"_"+input_count+".json",
    whitelist_file=whitelist_file,
    max_questions=None,
    max_new_tokens=2048,
)

print("Done", out_file)
