#%%
# Imports and Data Loading
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
import numpy as np

sns.set(style="whitegrid")

# Load the detailed analysis data
details_file_path = 'f_temp_check/outputs/mmlu/DeepSeek-R1-Distill-Llama-8B/sycophancy/temp_analysis_details_301.json'
summary_file_path = 'f_temp_check/outputs/mmlu/DeepSeek-R1-Distill-Llama-8B/sycophancy/temp_analysis_summary_301.json'

try:
    with open(details_file_path, 'r') as f:
        details_data = json.load(f)
    print(f"Successfully loaded details from: {details_file_path}")
except FileNotFoundError:
    print(f"Error: Details file not found at {details_file_path}")
    details_data = None
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {details_file_path}")
    details_data = None

try:
    with open(summary_file_path, 'r') as f:
        summary_data = json.load(f)
    print(f"Successfully loaded summary from: {summary_file_path}")
except FileNotFoundError:
    print(f"Error: Summary file not found at {summary_file_path}")
    summary_data = None
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {summary_file_path}")
    summary_data = None

#%%
# Hint Matching Analysis (Using Summary Data)

hint_match_proportions_true = []
hint_match_proportions_false = []

if summary_data and 'results_per_question_summary' in summary_data:
    for question_summary in summary_data['results_per_question_summary']:
        q_id = question_summary.get('question_id')
        original_verbalizes = question_summary.get('original_verbalizes_hint')
        counts = question_summary.get('aggregated_counts', {})
        
        attempted = counts.get('num_generations_attempted', 0)
        errors = counts.get('num_answer_na_or_error', 0)
        matched = counts.get('match_hint_count', 0)

        valid_generations = attempted - errors
        
        if valid_generations > 0:
            match_proportion = matched / valid_generations
        else:
            match_proportion = 0.0

        result_entry = {
            'question_id': q_id,
            'match_proportion': match_proportion,
            'match_count': matched,
            'valid_generations': valid_generations
        }

        if original_verbalizes:
            hint_match_proportions_true.append(result_entry)
        else:
            hint_match_proportions_false.append(result_entry)

    # Sort by match proportion (descending)
    hint_match_proportions_true.sort(key=lambda x: x['match_proportion'], reverse=True)
    hint_match_proportions_false.sort(key=lambda x: x['match_proportion'], reverse=True)

    print("\n--- Hint Match Proportion (Original Verbalizes Hint = True) ---")
    for entry in hint_match_proportions_true:
        print(f"QID: {entry['question_id']:<4} | Proportion: {entry['match_proportion']:.2f} ({entry['match_count']}/{entry['valid_generations']})")

    print("\n--- Hint Match Proportion (Original Verbalizes Hint = False) ---")
    for entry in hint_match_proportions_false:
        print(f"QID: {entry['question_id']:<4} | Proportion: {entry['match_proportion']:.2f} ({entry['match_count']}/{entry['valid_generations']})")

    # Simple bar-plot: top-N questions (verbalize=True) by match proportion
    top_n = 15
    top_df = pd.DataFrame(hint_match_proportions_true[:top_n])
    plt.figure(figsize=(10,4))
    sns.barplot(
        data=top_df,
        x="question_id", y="match_proportion", palette="Blues_d"
    )
    plt.title(f"Top {top_n} Qs (original_verbalize=True) – Hint-Match proportion")
    plt.ylim(0,1)
    plt.show()
else:
    print("\nCould not perform hint match analysis due to missing or invalid summary data.")

#%%
# Quartile Analysis for Matched & Verbalized Hints (Using Detailed Data)

quartile_counts_true = Counter()
total_matched_verbalized_true = 0
quartile_counts_false = Counter()
total_matched_verbalized_false = 0

if details_data and 'detailed_analysis' in details_data:
    for question_details in details_data['detailed_analysis']:
        original_verbalizes = question_details.get('original_verbalizes_hint')
        
        for gen_detail in question_details.get('generation_details', []):
            matched_hint = gen_detail.get('matched_hint_option', False)
            verification = gen_detail.get('verification_output', {})
            verbalized_hint = verification.get('verbalizes_hint', False)
            quartiles = verification.get('quartiles', [])

            if matched_hint and verbalized_hint:
                if original_verbalizes:
                    quartile_counts_true.update(quartiles)
                    total_matched_verbalized_true += 1
                else:
                    quartile_counts_false.update(quartiles)
                    total_matched_verbalized_false += 1

    print("\n--- Quartile Distribution (Matched & Verbalized Hint) ---")
    
    print("\nOriginal Verbalizes Hint = True:")
    if total_matched_verbalized_true > 0:
        print(f"  Total instances: {total_matched_verbalized_true}")
        for q in sorted(quartile_counts_true.keys()):
             count = quartile_counts_true[q]
             proportion = count / sum(quartile_counts_true.values()) # Proportion within this category
             print(f"  Quartile {q}: {count} ({proportion:.2%})")
    else:
        print("  No instances found where hint was matched and verbalized.")

    print("\nOriginal Verbalizes Hint = False:")
    if total_matched_verbalized_false > 0:
        print(f"  Total instances: {total_matched_verbalized_false}")
        for q in sorted(quartile_counts_false.keys()):
             count = quartile_counts_false[q]
             proportion = count / sum(quartile_counts_false.values()) # Proportion within this category
             print(f"  Quartile {q}: {count} ({proportion:.2%})")
    else:
        print("  No instances found where hint was matched and verbalized.")

 # Quartile distribution bar for matched & verbalized
    quart_df = pd.DataFrame({
        "quartile": sorted(set(list(quartile_counts_true)+list(quartile_counts_false))),
        "VerbalizesTrue": [quartile_counts_true[q] for q in sorted(quartile_counts_true)],
        "VerbalizesFalse": [quartile_counts_false[q] for q in sorted(quartile_counts_false)],
    }).melt(id_vars="quartile", var_name="group", value_name="count")

    plt.figure(figsize=(6,4))
    sns.barplot(data=quart_df, x="quartile", y="count", hue="group")
    plt.title("Quartile distribution – matched & verbalized")
    plt.show()

    # Joint quartile pair analysis (co-occurrence)
    joint_counts_true = Counter()
    joint_counts_false = Counter()

    # Re-iterate to collect pairs
    for question_details in details_data['detailed_analysis']:
        original_verbalizes = question_details.get('original_verbalizes_hint')
        for gen_detail in question_details.get('generation_details', []):
            matched_hint = gen_detail.get('matched_hint_option', False)
            verification = gen_detail.get('verification_output', {})
            verbalized_hint = verification.get('verbalizes_hint', False)
            quartiles = verification.get('quartiles', [])
            if matched_hint and verbalized_hint and len(quartiles) >= 2:
                # consider unique unordered pairs
                pairs = combinations(sorted(set(quartiles)), 2)
                if original_verbalizes:
                    joint_counts_true.update(pairs)
                else:
                    joint_counts_false.update(pairs)

    # Function to build a 4x4 matrix from Counter
    def build_joint_matrix(joint_counter):
        mat = np.zeros((4,4), dtype=int)
        for (q1, q2), cnt in joint_counter.items():
            mat[q1-1, q2-1] += cnt
            mat[q2-1, q1-1] += cnt  # symmetrical
        return mat

    joint_mat_true = build_joint_matrix(joint_counts_true)
    joint_mat_false = build_joint_matrix(joint_counts_false)

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)
    sns.heatmap(joint_mat_true, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Joint Quartile Co-occurrence (Verbalize=True)')
    axes[0].set_xlabel('Quartile'); axes[0].set_ylabel('Quartile')
    sns.heatmap(joint_mat_false, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Joint Quartile Co-occurrence (Verbalize=False)')
    axes[1].set_xlabel('Quartile');
    plt.tight_layout()
    plt.show()

else:
     print("\nCould not perform quartile analysis due to missing or invalid details data.")

#%%
# Summary of Findings

print("\n--- Exploratory Analysis Summary ---")
print("1. Hint Match Proportions:")
print("   - Calculated and sorted the proportion of valid generations matching the hint for each question.")
print("   - Separated results based on whether the original prompt verbalized the hint (True/False).")
print("   - This allows comparison of hint adherence between the two groups.")
if summary_data:
     true_avg = summary_data.get("overall_summary", {}).get("group_original_verbalize_true", {}).get("avg_match_hint_proportion", "N/A")
     false_avg = summary_data.get("overall_summary", {}).get("group_original_verbalize_false", {}).get("avg_match_hint_proportion", "N/A")
     print(f"   - Average hint match proportion (Original Verbalizes True): {true_avg:.2f}" if isinstance(true_avg, float) else f"   - Average hint match proportion (Original Verbalizes True): {true_avg}")
     print(f"   - Average hint match proportion (Original Verbalizes False): {false_avg:.2f}" if isinstance(false_avg, float) else f"   - Average hint match proportion (Original Verbalizes False): {false_avg}")


print("\n2. Quartile Distribution (Matched & Verbalized):")
print("   - Analyzed the distribution of quartiles (1-4) where the generation *both* matched the hint and verbalized it.")
print("   - Counted occurrences within each quartile, separated by the 'original_verbalizes_hint' category.")
print("   - This helps understand *when* in the reasoning process the verbalization tends to occur for successful hint adherence.")
if details_data:
    print("   - Absolute counts and percentages within each category (True/False) are provided.")
    # Potential further insight: Compare the *shape* of the distributions.
    # E.g., Does one group verbalize earlier (Q1/Q2) more often than the other?
    # This requires interpreting the printed distributions above.
else:
     print("   - Analysis skipped due to data loading issues.")

print("\nNote: Analysis based on the provided data files.")

# %%
