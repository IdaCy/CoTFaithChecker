#!/usr/bin/env python
# coding: utf-8

# In[1]:

from datetime import datetime, timezone
import socket, os, sys

start_ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
print(
    f"\nSegmentation-annotate job started {start_ts} "
    f"on host {socket.gethostname()} (PID {os.getpid()}) ===",
    flush=True,
)

import os
from pathlib import Path

print("Using os.getcwd():", os.getcwd())
print("Using Path.cwd():",  Path.cwd())

print("\nDirectory contents:")
for p in Path.cwd().iterdir():
    print("  ", p.name)

# move two directories up from the location of this file
project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)
print("Working directory set to:", os.getcwd())

"""
nohup python -u c_cluster_analysis/segmentation/segment_annotate_pyversion.py > logs/segm_annotate_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python -u -m c_cluster_analysis.segmentation.segment_annotate_pyversion \
      > logs/segm_annotate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""


# In[3]:


import google.generativeai as genai
from google.generativeai import types

from c_cluster_analysis.segmentation.llm_annotator_confidence import run_annotation_pipeline

q_file = "data/mmlu/input_mcq_data.json"
api_key = "AIzaSyAA7FRVBJwtSrpMZQZVkzzIVaCqHrabrKo"
model_name = "gemini-2.5-flash-preview-04-17"

# from
from_base = "data/mmlu/DeepSeek-R1-Distill-Llama-8B/"
from_file = "/completions_with_1001.json"

# to
to_base = "c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B/annotations_confidence_"
to_file = ".json"

### NONE
htype = "none"
print(
    f"\nstarting {htype} "
    f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}"
)
run_annotation_pipeline(
    completions_file = from_base + htype + from_file,
    questions_file = q_file,
    output_file = to_base + htype + to_file,
    api_key=api_key,
    model_name = model_name,
    max_items=1,
)

### Sycophancy
htype = "sycophancy"
print(
    f"\nstarting {htype} "
    f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}"
)
run_annotation_pipeline(
    completions_file = from_base + htype + from_file,
    questions_file = q_file,
    output_file = to_base + htype + to_file,
    api_key=api_key,
    model_name = model_name,
    max_items=1,
)

### URGENCY
from_file = "/completions_with_1000.json"
htype = "induced_urgency"
print(
    f"\nstarting {htype} "
    f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}"
)
run_annotation_pipeline(
    completions_file = from_base + htype + from_file,
    questions_file = q_file,
    output_file = to_base + htype + to_file,
    api_key=api_key,
    model_name = model_name,
    max_items=1,
)

### UNETH
from_file = "/completions_with_500.json"
htype = "unethical_information"
print(
    f"\nstarting {htype} "
    f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}"
)
run_annotation_pipeline(
    completions_file = from_base + htype + from_file,
    questions_file = q_file,
    output_file = to_base + htype + to_file,
    api_key=api_key,
    model_name = model_name,
    max_items=1,
)
print(
    f"\finishing {htype} "
    f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}"
)
