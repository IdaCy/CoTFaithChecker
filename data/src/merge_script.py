import json
import os
from pathlib import Path

# Define paths and file patterns
base_dir = Path("data/mmlu/DeepSeek-R1-Distill-Llama-8B/none")
file_types = ["completions", "verification"]
input_suffixes = ["_with_2001.json", "_with_3000.json"]
output_suffix = "_with_5001.json"

# Ensure the base directory exists
if not base_dir.is_dir():
    print(f"Error: Base directory not found: {base_dir}")
    exit(1)

# Process each file type
for file_type in file_types:
    print(f"Processing type: {file_type}...")
    merged_data = []
    found_files = False

    for suffix in input_suffixes:
        input_filename = f"{file_type}{suffix}"
        input_filepath = base_dir / input_filename

        if input_filepath.is_file():
            found_files = True
            print(f"  Loading {input_filepath}...")
            try:
                with open(input_filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        print(f"  Warning: Expected a list in {input_filepath}, got {type(data)}. Skipping.")
            except json.JSONDecodeError:
                print(f"  Error: Could not decode JSON from {input_filepath}. Skipping.")
            except Exception as e:
                print(f"  Error loading {input_filepath}: {e}")
        else:
            print(f"  Warning: File not found: {input_filepath}")

    if not found_files:
        print(f"  No input files found for type {file_type}. Skipping merge.")
        continue

    if not merged_data:
        print(f"  No data loaded for type {file_type}. Skipping output.")
        continue

    # Define output file path
    output_filename = f"{file_type}{output_suffix}"
    output_filepath = base_dir / output_filename

    # Write merged data to output file
    print(f"  Saving merged data to {output_filepath} ({len(merged_data)} items)...")
    try:
        with open(output_filepath, 'w') as f:
            json.dump(merged_data, f, indent=2) # Use indent for readability
        print(f"  Successfully saved {output_filepath}")
    except Exception as e:
        print(f"  Error writing to {output_filepath}: {e}")

print("Merge process completed.")
