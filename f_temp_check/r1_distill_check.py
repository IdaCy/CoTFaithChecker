# %%
# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
# Define model name, chat template, and instruction
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
instruction = "what is 5*5 +123432?"

# %%
# Load tokenizer and model
# Using bfloat16 and device_map='auto' for potential performance gains on capable GPUs
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# %%
# Format the prompt
prompt = CHAT_TEMPLATE.format(instruction=instruction)

# %%
# Tokenize the input
# Ensure tensors are moved to the same device as the model
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# %%
# Generate completion
# Set pad_token_id to eos_token_id for open-ended generation
outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

# %%
# Decode the output, preserving special tokens as requested
# We take outputs[0] as generate usually returns a batch
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

# %%
# Print the full generated text including the prompt and special tokens
print("Generated Text:")
print(decoded_output)
