Working dir: /root/CoTFaithChecker
CoTFaithChecker/e_confirm_xy_yx_5_nohup.py starting!
[2025-04-27T21:49:35.692182] starting setup
[2025-04-27T21:49:36.092048] loading model & tokenizer
2025-04-27 21:49:37,707 - INFO - CUDA is available. Using GPU.
2025-04-27 21:49:37,707 - INFO - Loading model and tokenizer: deepseek-ai/DeepSeek-R1-Distill-Llama-8B onto cuda
/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:817: UserWarning: `return_dict_in_generate` is NOT set to `True`, but `output_hidden_states` is. When `return_dict_in_generate` is not `True`, `output_hidden_states` is ignored.
  warnings.warn(
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:01<00:00,  1.63it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:01<00:00,  1.62it/s]
2025-04-27 21:49:42,230 - INFO - Model and tokenizer loaded successfully.
data_loader — INFO — Logger initialised; log file = /root/CoTFaithChecker/logs/data_loader_20250427_214942.log
2025-04-27 21:49:42,233 - INFO - Logger initialised; log file = /root/CoTFaithChecker/logs/data_loader_20250427_214942.log
[2025-04-27T21:49:42.234325] loading data
data_loader — INFO — → kept 5 after cluster filter ['arts']
2025-04-27 21:49:42,234 - INFO - → kept 5 after cluster filter ['arts']
data_loader — INFO — Found 5 files in data/chainscope/questions_json/gt_NO_1
2025-04-27 21:49:42,234 - INFO - Found 5 files in data/chainscope/questions_json/gt_NO_1
data_loader — INFO — Total files collected: 5
2025-04-27 21:49:42,234 - INFO - Total files collected: 5
prompt_builder — INFO — Logger initialised; log file = /root/CoTFaithChecker/logs/prompt_builder_20250427_214942.log
2025-04-27 21:49:42,235 - INFO - Logger initialised; log file = /root/CoTFaithChecker/logs/prompt_builder_20250427_214942.log
prompt_builder — INFO — PromptBuilder initialised with style=instr-v0, mode=cot
2025-04-27 21:49:42,235 - INFO - PromptBuilder initialised with style=instr-v0, mode=cot
[2025-04-27T21:49:42.235520] running inference
inference — INFO — Logger initialised; log file = /root/CoTFaithChecker/logs/inference_20250427_214942.log
2025-04-27 21:49:42,235 - INFO - Logger initialised; log file = /root/CoTFaithChecker/logs/inference_20250427_214942.log
inference — INFO — Processing wm-book-length_gt_NO_1_6fda02e3.json
2025-04-27 21:49:42,236 - INFO - Processing wm-book-length_gt_NO_1_6fda02e3.json
2025-04-27 21:49:42,236 - DEBUG - Loading data/chainscope/questions_json/gt_NO_1/wm-book-length_gt_NO_1_6fda02e3.json
2025-04-27 21:49:42,236 - DEBUG -   ↳ run 1/1
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Traceback (most recent call last):
  File "/root/CoTFaithChecker/h_hidden_states/xyyx/run_xyyx.py", line 136, in <module>
    run_inference(
  File "/root/CoTFaithChecker/e_confirm_xy_yx/main/inference.py", line 188, in run_inference
    out_path = _process_single_file(
  File "/root/CoTFaithChecker/e_confirm_xy_yx/main/inference.py", line 117, in _process_single_file
    gen_out = _generate(
  File "/root/CoTFaithChecker/e_confirm_xy_yx/main/inference.py", line 59, in _generate
    outputs = model.generate(**encodings, **gen_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 2326, in generate
    result = self._sample(
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 3289, in _sample
    outputs = model_forward(**model_inputs, return_dict=True)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py", line 853, in forward
    outputs = self.model(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py", line 601, in forward
    layer_outputs = decoder_layer(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py", line 343, in forward
    hidden_states, self_attn_weights = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py", line 299, in forward
    attn_output, attn_weights = attention_interface(
  File "/usr/local/lib/python3.10/dist-packages/transformers/integrations/sdpa_attention.py", line 54, in sdpa_attention_forward
    attn_output = torch.nn.functional.scaled_dot_product_attention(
KeyboardInterrupt
