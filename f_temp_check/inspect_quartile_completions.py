#%%

%cd ..

#%%
"""
Inspect and print completions where the hint is verbalized in specific quartiles.

We will load:
 1. `temp_analysis_details_301.json`  (detailed LLM-verifier outputs)
 2. Corresponding `temp_generations_raw_*.json` (raw completions)

We then extract:
  • All completions whose `verification_output.quartiles` list contains 1  → **quartile_1_set**
  • All completions whose quartiles contain both 1 **and** 4            → **quartile_1_and_4_set**

Finally, we pretty-print the completions (qid, run_index) for each subset.

Note: This script ignores the `original_verbalizes_hint` flag – we consider all generations.
"""

#%%
# Imports & file paths
import json
from pathlib import Path

base_dir = Path('f_temp_check/outputs/mmlu/DeepSeek-R1-Distill-Llama-8B/sycophancy')

details_path = base_dir / 'temp_analysis_details_301.json'
raw_path     = base_dir / 'temp_generations_raw_mmlu_301.json'

print(f"Details path: {details_path}\nRaw path:     {raw_path}")

#%%
# Load JSON files
with open(details_path, 'r') as f:
    details_data = json.load(f)

with open(raw_path, 'r') as f:
    raw_data = json.load(f)

# Build quick lookup: (qid, run_index) -> completion text
raw_lookup = {}
for q_entry in raw_data['raw_generations']:
    qid = q_entry['question_id']
    for idx, completion in enumerate(q_entry['generations']):
        raw_lookup[(qid, idx)] = completion

print(f"Loaded {len(raw_lookup):,} raw completions.")

#%%
# Collect generation keys for desired quartile patterns
quartile1_keys = []
quartile1_and4_keys = []

detailed_list = details_data['detailed_analysis']
for q_entry in detailed_list:
    qid = q_entry['question_id']
    for gen in q_entry['generation_details']:
        idx = gen['run_index']
        verification = gen.get('verification_output', {})
        quartiles = verification.get('quartiles', [])
        if not quartiles:
            continue
        if 1 in quartiles:
            quartile1_keys.append((qid, idx))
            if 4 in quartiles:
                quartile1_and4_keys.append((qid, idx))

print(f"Found {len(quartile1_keys)} completions with quartile 1 (any additional).")
print(f"Found {len(quartile1_and4_keys)} completions with quartiles 1 & 4.")

#%%
# Helper to pretty-print a subset

def print_subset(keys, title, max_chars=None):
    print("\n" + "="*90)
    print(title)
    print("="*90)
    
    start_marker = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<think>"
    end_marker = "</think>"
    
    for (qid, idx) in keys:
        comp_text = raw_lookup.get((qid, idx), "<MISSING COMPLETION>")
        header = f"[QID {qid} | run {idx}]" + "-"*40
        print(header)
        
        start_index = comp_text.rfind(start_marker)
        if start_index != -1:
            # Look for end marker *after* the last start marker
            end_index = comp_text.find(end_marker, start_index + len(start_marker))
            if end_index != -1:
                # Found both: extract text between them
                extracted_content = comp_text[start_index + len(start_marker):end_index].strip()
            else:
                # Start found, end missing: extract from start to end of string
                extracted_content = comp_text[start_index + len(start_marker):].strip()
                extracted_content += "\n[NOTE: End marker '</think>' was missing]"
                
            # Optionally truncate extracted content if needed
            if max_chars and len(extracted_content) > max_chars:
                 extracted_content = extracted_content[:max_chars] + "... [TRUNCATED CONTENT]"
            print(extracted_content)

        else:
            # Start marker not found
            print("[NOTE: Start marker '...<think>' not found]")
            
        print("\n")

#%%
# Print results
print_subset(quartile1_keys, "COMPLETIONS WITH HINT IN QUARTILE 1")
print_subset(quartile1_and4_keys, "COMPLETIONS WITH HINT IN QUARTILES 1 AND 4")

#%%
print("Done.") 
# %%
