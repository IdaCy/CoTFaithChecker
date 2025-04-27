import os
import json
from datasets import load_dataset

def main():
    # Define the output directory and file path
    output_dir = "mmlu_pro_math"
    output_file = os.path.join(output_dir, "input_mcq_data.json")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test") # Assuming test split, adjust if needed

    # Filter for math category
    print("Filtering for math category...")
    math_data = dataset.filter(lambda example: example['category'] == 'math')

    # Transform the data
    print("Transforming data...")
    transformed_data = []
    for example in math_data:
        transformed_entry = {
            "question_id": example["question_id"],
            "question": example["question"],
            "correct": example["answer"], # Map answer to correct
        }

        # Map options list to A, B, C, ... keys
        for i, option_text in enumerate(example["options"]):
            option_key = chr(ord('A') + i)
            transformed_entry[option_key] = option_text

        transformed_data.append(transformed_entry)

    # Save the transformed data to JSON
    print(f"Saving transformed data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=2)

    print("Data processing complete.")

if __name__ == "__main__":
    main()



