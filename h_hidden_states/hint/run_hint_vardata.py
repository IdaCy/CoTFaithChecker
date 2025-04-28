#!/usr/bin/env python
# coding: utf-8


import sys, pathlib, os, logging, json
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))          # <── key line
os.chdir(PROJECT_ROOT)
print("Working dir:", PROJECT_ROOT)

from h_hidden_states.hint.pipeline import generate_dataset_completions
from h_hidden_states.hint.model_handler import load_model_and_tokenizer

#model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model, tokenizer, model_name, device = load_model_and_tokenizer(model_path)

hint_types = ["unethical_information", "induced_urgency"]
#hint_types = ["none", "sycophancy", "unethical_information", "induced_urgency"]
#hint_types = ["sycophancy"]
n_questions = 500


### MMLU - original

dataset_name = "mmlu"

generate_dataset_completions(
    model = model,
    tokenizer = tokenizer,
    model_name = model_name,
    device = device,
    dataset_name = dataset_name,
    hint_types = hint_types,
    batch_size = 32,
    max_new_tokens = None, 
    n_questions = n_questions,
    output_location = "h_hidden_states/outputs/hints/" + dataset_name + "/" + model_name + "/" + str(n_questions) + "_captures",
)

### re-defining hint types so that all used later on
hint_types = ["none", "sycophancy", "unethical_information", "induced_urgency"]


### MMLU Pro Maths

dataset_name = "mmlu_pro_maths"

generate_dataset_completions(
    model = model,
    tokenizer = tokenizer,
    model_name = model_name,
    device = device,
    dataset_name = dataset_name,
    hint_types = hint_types,
    batch_size = 32,
    max_new_tokens = None, 
    n_questions = n_questions,
    output_location = "h_hidden_states/outputs/hints/" + dataset_name + "/" + model_name + "/" + str(n_questions) + "_captures",    
)

### GSM8K - original

dataset_name = "gsm8k"

generate_dataset_completions(
    model = model,
    tokenizer = tokenizer,
    model_name = model_name,
    device = device,
    dataset_name = dataset_name,
    hint_types = hint_types,
    batch_size = 32,
    max_new_tokens = None,  
    n_questions = n_questions,
    output_location = "h_hidden_states/outputs/hints/" + dataset_name + "/" + model_name + "/" + str(n_questions) + "_captures",
)


### MMLU Pro

dataset_name = "mmlu_pro"

generate_dataset_completions(
    model = model,
    tokenizer = tokenizer,
    model_name = model_name,
    device = device,
    dataset_name = dataset_name,
    hint_types = hint_types,
    batch_size = 32,
    max_new_tokens = None, 
    n_questions = n_questions,  
    output_location = "h_hidden_states/outputs/hints/" + dataset_name + "/" + model_name + "/" + str(n_questions) + "_captures",
)

