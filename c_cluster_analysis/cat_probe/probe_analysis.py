# ──────────────────────────────────────────────────────────────────────────────
#  0. Imports & configuration
# ──────────────────────────────────────────────────────────────────────────────
import json, numpy as np, torch, sklearn.linear_model as sklin
from pathlib import Path
CATEGORY_NAMES = [
    "problem_restating", "knowledge_augmentation", "assumption_validation",
    "logical_deduction", "option_elimination",
    "uncertainty_or_certainty_expression", "backtracking", "forward_planning",
    "decision_confirmation", "answer_reporting", "option_restating", "other"
]
HIDDEN_LAYER = "layer_32"

# ──────────────────────────────────────────────────────────────────────────────
#  1. Load annotation labels
# ──────────────────────────────────────────────────────────────────────────────
with open("annotations.json", encoding="utf-8") as f:
    ann_raw = json.load(f)

# build (qid, sid) → category-scores dict
ann_map = {}
for q in ann_raw:
    qid = q["question_id"]
    for sent in q["annotations"]:
        ann_map[(qid, sent["sentence_id"])] = sent

# ──────────────────────────────────────────────────────────────────────────────
#  2. Load hidden-state captures
# ──────────────────────────────────────────────────────────────────────────────
rows, labels = [], []
with open("sentence_level_hidden.json", encoding="utf-8") as f:
    data = json.load(f)

for rec in data:
    qid = rec["question_id"]
    for sent in rec["sentences"]:
        sid   = sent["sentence_id"]
        hs    = np.array(sent["pooled_hs"][HIDDEN_LAYER], dtype=np.float32)
        ann   = ann_map[(qid, sid)]
        cat_idx = int(np.argmax([ann[c] for c in CATEGORY_NAMES]))
        rows.append(hs)
        labels.append(cat_idx)

X = np.vstack(rows)
y = np.array(labels)
print("dataset shape:", X.shape, "labels:", y.shape)

# ──────────────────────────────────────────────────────────────────────────────
#  3. Train logistic-regression probe
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=42)

probe = sklin.LogisticRegression(
    max_iter=2000, multi_class="ovr", solver="lbfgs", n_jobs=-1)
probe.fit(X_train, y_train)
print("dev accuracy:", probe.score(X_dev, y_dev))

# keep coefficients – shape: (n_labels, hidden_size)
coef = probe.coef_.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
#  4. Compute source→target steering direction
# ──────────────────────────────────────────────────────────────────────────────
SOURCE = CATEGORY_NAMES.index("backtracking")
TARGET = CATEGORY_NAMES.index("logical_deduction")

direction = coef[TARGET] - coef[SOURCE]
direction = direction / np.linalg.norm(direction)
direction_t = torch.tensor(direction, dtype=torch.float32)

# save for later
np.save("steer_direction.npy", direction)

# ──────────────────────────────────────────────────────────────────────────────
#  5. Hook for residual-stream steering
# ──────────────────────────────────────────────────────────────────────────────
import transformers, types

class ResidualSteerHook(torch.nn.Module):
    def __init__(self, direction: torch.Tensor, scale: float = 1.0):
        super().__init__()
        self.register_buffer("d", direction)
        self.scale = scale
    def forward(self, hidden, *rest, **kw):
        # hidden: (batch, seq, hidden_size)
        hidden = hidden + self.scale * self.d
        return (hidden, *rest)

def insert_steer_hook(model, layer_idx: int, direction: torch.Tensor, scale=1.0):
    """
    Adds a forward hook to `model.model.layers[layer_idx]` that nudges the
    residual stream towards `direction` by `scale`.
    Returns a handle so you can .remove() it later.
    """
    layer = model.model.layers[layer_idx]
    return layer.register_forward_hook(
        lambda mod, inp, out: ResidualSteerHook(direction, scale)(out[0], *out[1:])
    )

# ──────────────────────────────────────────────────────────────────────────────
#  6. Quick manual test
# ──────────────────────────────────────────────────────────────────────────────
# from transformers import AutoModelForCausalLM, AutoTokenizer
# m, t = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-chat-hf",
#                                             torch_dtype=torch.float16,
#                                             device_map="auto"), \
#        AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-chat-hf")
#
# hook_handle = insert_steer_hook(m, layer_idx=31, direction=direction_t, scale=2.0)
# out = m.generate(**t("Explain gravitational lensing.", return_tensors="pt" ).to(m.device),
#                  max_new_tokens=128)
# print(t.decode(out[0], skip_special_tokens=True))
# hook_handle.remove()
