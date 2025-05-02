#%%
%cd ..

#%% Imports and configuration
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Experiment metadata
n_questions = 301
model_name = "DeepSeek-R1-Distill-Llama-8B"
dataset_name = "mmlu"
hint_type = "sycophancy"

#%% Load analysis summary JSON
# Build the path to the summary file (relative to repository root)
data_path = (
    Path("f_temp_check")
    / "outputs"
    / dataset_name
    / model_name
    / hint_type
    / f"temp_analysis_summary_{n_questions}.json"
)

with data_path.open() as f:
    analysis = json.load(f)

# Prepare a DataFrame with one row per question
records = []
for q in analysis["results_per_question_summary"]:
    counts = q["aggregated_counts"]
    match_cnt = counts["match_hint_count"]
    # Skip questions where the model never matched the hint (division by zero)
    if match_cnt == 0:
        continue

    cond_prop = counts["match_and_verbalize_count"] / match_cnt
    records.append(
        {
            "question_id": q["question_id"],
            "original_verbalizes_hint": (
                "Verbalized" if q["original_verbalizes_hint"] else "Did not verbalize"
            ),
            "conditional_verbalize_given_match_proportion": cond_prop,
        }
    )

df = pd.DataFrame(records)

#%% Violin plot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

ax = sns.violinplot(
    data=df,
    x="original_verbalizes_hint",
    y="conditional_verbalize_given_match_proportion",
    hue="original_verbalizes_hint",
    palette="Set2",
    dodge=False,  # keep violins centered on their category
    # inner="quartile",
    cut=0,
)

# Remove duplicate legend
# ax.legend_.remove()

ax.set_title("Distribution of P(Verbalize | Match) across Questions")
ax.set_xlabel("Original rollout behaviour")
ax.set_ylabel("Hint Verbalization (%) Across N Rollouts")
plt.tight_layout()
plt.show()

#%% Unconditional verbalization proportion per question
records_uncond = []
for q in analysis["results_per_question_summary"]:
    counts = q["aggregated_counts"]
    total_gen = counts["num_generations_analyzed_for_verbalization"]
    if total_gen == 0:
        continue
    uncond_prop = counts["verbalize_hint_count"] / total_gen
    records_uncond.append(
        {
            "question_id": q["question_id"],
            "original_verbalizes_hint": (
                "Verbalized" if q["original_verbalizes_hint"] else "Did not verbalize"
            ),
            "unconditional_verbalize_proportion": uncond_prop,
        }
    )

df_uncond = pd.DataFrame(records_uncond)

#%% Violin plot (unconditional)
plt.figure(figsize=(8, 6))
ax2 = sns.violinplot(
    data=df_uncond,
    x="original_verbalizes_hint",
    y="unconditional_verbalize_proportion",
    hue="original_verbalizes_hint",
    palette="Set2",
    dodge=False,
    # inner="quartile",
    cut=0,
)
# ax2.legend_.remove()
ax2.set_title("Distribution of P(Verbalize) across Questions")
ax2.set_xlabel("Original rollout behaviour")
ax2.set_ylabel("Hint Verbalization (%) Across N Rollouts")
plt.tight_layout()
plt.show()

# %%
