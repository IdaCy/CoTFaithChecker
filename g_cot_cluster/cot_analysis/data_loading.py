
import json, itertools, pandas as pd, pathlib, re
from typing import List, Dict, Any, Tuple

def load_segmented_json(path: str, hint_type: str) -> pd.DataFrame:
    """Read one segmented_*.json file → DataFrame with one row per question."""
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    rows = []
    for q in data:
        seq = [seg['phrase_category'] for seg in q['segments']]
        rows.append({
            'question_id': q.get('question_id'),
            'segments': q['segments'],
            'category_sequence': seq,
            'hint_type': hint_type
        })
    return pd.DataFrame(rows)

def load_segmented_jsons(file_map: Dict[str, str]) -> pd.DataFrame:
    """file_map: {hint_type: path}. Returns concatenated DataFrame."""
    df_list = [load_segmented_json(p, h) for h, p in file_map.items()]
    return pd.concat(df_list, ignore_index=True)

def build_category_sequences(df: pd.DataFrame) -> pd.Series:
    """Return category_sequence column as Series of tuples for hashing."""
    return df['category_sequence'].apply(tuple)
