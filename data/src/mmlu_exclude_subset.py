import os
import json
from datasets import load_dataset

def exclude_and_save_mmlu(subset_path="data/mmlu/input_mcq_data.json", 
                          output_dir="data/mmlu_new", 
                          output_filename="input_mcq_data.json"):
    """
    Loads the full MMLU dataset, excludes questions found in a specified subset JSON file,
    formats the remaining questions, and saves them to a new JSON file.

    Args:
        subset_path (str): Path to the JSON file containing the subset of questions to exclude.
        output_dir (str): The directory to save the output JSON file.
        output_filename (str): The name of the output JSON file.
    """
    print(f"Loading subset questions from {subset_path}...")
    try:
        with open(subset_path, 'r') as f:
            subset_data = json.load(f)
        # Store the 'question' text from the subset for quick lookup
        subset_questions = {item['question'] for item in subset_data}
        print(f"Loaded {len(subset_questions)} questions to exclude.")
    except FileNotFoundError:
        print(f"Error: Subset file not found at {subset_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {subset_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the subset file: {e}")
        return

    print("Loading full MMLU dataset (all subjects)...")
    try:
        # Load the 'test' split which contains the questions and answers
        mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
        print(f"Full dataset loaded successfully. Total examples: {len(mmlu_dataset)}")
    except Exception as e:
        print(f"Error loading full dataset: {e}")
        return

    print("Filtering and formatting remaining questions...")
    formatted_data = []
    new_question_id = 0
    excluded_count = 0
    included_count = 0

    for example in mmlu_dataset:
        question_text = example['question']
        
        # Check if this question is in the subset to be excluded
        if question_text in subset_questions:
            excluded_count += 1
            continue # Skip this question

        # Process and format the question if it's not in the subset
        correct_answer_index = example['answer']
        choices = example['choices']
        choice_labels = [chr(ord('A') + j) for j in range(len(choices))]

        if 0 <= correct_answer_index < len(choice_labels):
            correct_label = choice_labels[correct_answer_index]
        else:
            # Handle potential invalid index (though unlikely for standard MMLU)
            print(f"Warning: Invalid answer index {correct_answer_index} found in full dataset for question: {question_text[:50]}... Skipping.")
            continue

        formatted_entry = {
            "question_id": new_question_id + 20000,
            "question": question_text,
            "correct": correct_label,
            "subject": example['subject'],
        }
        
        for label, choice_text in zip(choice_labels, choices):
            formatted_entry[label] = choice_text
            
        formatted_data.append(formatted_entry)
        new_question_id += 1
        included_count += 1

    print(f"Filtering complete. Included {included_count} questions, excluded {excluded_count} questions.")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving formatted data ({len(formatted_data)} items) to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        print("Data saved successfully.")
    except IOError as e:
        print(f"Error saving data to file: {e}")

if __name__ == "__main__":
    exclude_and_save_mmlu() 