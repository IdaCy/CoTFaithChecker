#%%
%cd ..

#%%
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
inputs = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "Hello, how are you?"},
    ],
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))


# %%

# 1. Create a batch of single-turn conversations
conversations_batch = [
    [{"role": "user", "content": "Hello, how are you?"}], # First conversation (single turn)
    [{"role": "user", "content": "What is the capital of France?"}], # Second conversation (single turn)
    [{"role": "user", "content": "Explain the theory of relativity in simple terms."}] # Third conversation (single turn)
]

# Apply template to get strings (tokenize=False)
# We need add_generation_prompt=True for each item
formatted_prompts = [
    tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    for conv in conversations_batch
]

# 2. Set padding strategy
# For generation, left padding is usually preferred
tokenizer.padding_side = "left"
# If the tokenizer doesn't have a pad token, set it to EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenize the batch with padding
inputs = tokenizer(
    formatted_prompts,
    return_tensors="pt",
    padding=True,
).to("cuda")


# 4. Generate responses for the batch
# inputs is already a dictionary containing input_ids and attention_mask
outputs = model.generate(**inputs, max_new_tokens=300)

# 5. Decode the outputs
# Use batch_decode for multiple sequences
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(decoded_outputs):
    print(f"--- Output {i+1} ---")
    print(output)
    print("-" * (15 + len(str(i+1))))


# %%
print(tokenizer.chat_template)
# %%
