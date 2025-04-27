# utils/data.py
from pathlib import Path
import json, torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModel.from_pretrained(model_name,
                                      output_hidden_states=True).to(DEVICE).eval()

def load_segments(seg_file):
    """Yield (question_id, phrase_category, char_start, char_end, text)."""
    for ex in json.loads(Path(seg_file).read_text()):
        qid = ex["question_id"]
        for seg in ex["segments"]:
            yield (qid,
                   seg["phrase_category"],
                   seg["start"],
                   seg["end"],
                   seg["text"])

def segment_hidden_state(full_completion:str,
                         seg_start:int,
                         seg_end:int,
                         layer:int=-1):
    tokens = tokenizer(full_completion, return_offsets_mapping=True,
                       return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outs = model(**tokens)
    hs = outs.hidden_states[layer][0]           # (seq_len, hidden_dim)
    # mask for tokens whose span overlaps [seg_start, seg_end]
    mask = [(s < seg_end) and (e > seg_start)
            for s,e in tokens.offset_mapping[0].tolist()]
    seg_vec = hs[mask].mean(0)                  # (hidden_dim,)
    return seg_vec.cpu()


