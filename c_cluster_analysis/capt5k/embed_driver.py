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

LOG_FILE = "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(process)d - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w")]
)

accelerator = Accelerator()
if accelerator.is_main_process:
    logging.getLogger().addHandler(logging.StreamHandler())

print(
    f"on host {socket.gethostname()} (PID {os.getpid()}) ===",
)

print("starting at", datetime.now(ZoneInfo("Europe/London")).isoformat(timespec="seconds"))

from h_hidden_space.capt5k.embeddings_pipeline import generate_dataset_embeddings, EmbeddingModel
from h_hidden_space.capt5k.model_handler import load_model_and_tokenizer
from accelerate import Accelerator

accelerator = Accelerator()

model, tokenizer, model_name, device = load_model_and_tokenizer("/path/to/llama-8b")

embedder = EmbeddingModel(                         # local, fast, no API key
    model_name="intfloat/e5-large-v2",
    device=device
)

generate_dataset_embeddings(
    accelerator=accelerator,
    model=model,
    tokenizer=tokenizer,
    model_name=model_name,
    device=device,
    dataset_name="stat_mcq",
    hint_types=["none", "verbalized", "unverbalized"],
    batch_size=32,
    max_new_tokens=2048,
    n_questions=None,
    embedder=embedder,
    sentence_level=True,        # False to embed whole CoT
)
