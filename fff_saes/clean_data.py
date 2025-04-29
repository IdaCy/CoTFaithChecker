import json
import os
import re

def load_json(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def load_completions_by_id(file_path):
    """Loads completions and organizes them by question_id."""
    data = load_json(file_path)
    if data is None:
        return None
    completions_dict = {item['question_id']: item['completion'] for item in data}
    return completions_dict

def clean_hinted_completion(completion):
    """Removes text between specified markers in the completion string."""
    start_marker = "Please answer with the letter of the corresponding to the correct option."
    end_marker = "<|eot_id|>"

    start_index = completion.find(start_marker)
    end_index = completion.find(end_marker)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        # Keep the start marker, remove text after it until the end marker
        text_to_remove = completion[start_index + len(start_marker):end_index]
        cleaned_completion = completion.replace(text_to_remove, '', 1) # Replace only the first occurrence
        return cleaned_completion
    else:
        # If markers not found or in wrong order, return original (or handle as error)
        # print(f"Warning: Markers not found or in wrong order for cleaning completion. Start: {start_index}, End: {end_index}")
        return completion # Return original for now

def process_data(model: str, dataset: str, hint_type: str, num_questions: str):
    """Loads, filters, cleans, and saves completion data."""

    # Construct file paths
    base_data_path = 'data'
    verification_file = os.path.join(base_data_path, dataset, model, hint_type, f'hint_verification_with_{num_questions}.json')
    hinted_completions_file = os.path.join(base_data_path, dataset, model, hint_type, f'completions_with_{num_questions}.json')
    unhinted_completions_file = os.path.join(base_data_path, dataset, model, 'none', f'completions_with_{num_questions}.json')

    # Load verification data
    verification_data = load_json(verification_file)
    if verification_data is None:
        return

    # Load completions
    hinted_completions = load_completions_by_id(hinted_completions_file)
    if hinted_completions is None:
        return
    unhinted_completions = load_completions_by_id(unhinted_completions_file)
    if unhinted_completions is None:
        return

    # Filter and process data
    output_data = []
    processed_ids = set()

    for item in verification_data:
        question_id = item.get('question_id')
        verbalizes_hint = item.get('verbalizes_hint')

        if question_id is None or verbalizes_hint is None:
            print(f"Warning: Skipping verification item due to missing 'question_id' or 'verbalizes_hint': {item}")
            continue

        if question_id in processed_ids:
             print(f"Warning: Duplicate question_id {question_id} found in verification data. Skipping.")
             continue

        if not verbalizes_hint:
            hinted_comp = hinted_completions.get(question_id)
            unhinted_comp = unhinted_completions.get(question_id)

            if hinted_comp is None:
                print(f"Warning: Hinted completion not found for question_id {question_id}. Skipping.")
                continue
            if unhinted_comp is None:
                print(f"Warning: Unhinted completion not found for question_id {question_id}. Skipping.")
                continue

            cleaned_hinted_comp = clean_hinted_completion(hinted_comp)

            output_data.append({
                'question_id': question_id,
                'hinted_completion': cleaned_hinted_comp,
                'unhinted_completion': unhinted_comp
            })
            processed_ids.add(question_id)


    if not output_data:
        print("No questions found where verbalizes_hint is False. No output generated.")
        return

    # Construct output path and save data
    output_dir = os.path.join('fff_saes', 'data', 'input', dataset, model, hint_type)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{num_questions}.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully processed {len(output_data)} items.")
        print(f"Output saved to {output_file}")
    except IOError as e:
        print(f"Error writing output file {output_file}: {e}")

if __name__ == '__main__':
    # Example usage:
    process_data(
        model="DeepSeek-R1-Distill-Llama-8B",
        dataset="mmlu",
        hint_type="sycophancy",
        num_questions="1001"
    )

