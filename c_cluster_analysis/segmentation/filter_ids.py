from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Set

def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_verification_only_ids(hint_data: list[dict]) -> Set[int]:
    return {
        rec["question_id"]
        for rec in hint_data
        if rec.get("verbalizes_hint") is False
    }


def filter_ids(completions: str, hint_verification: str, output_file: str) -> None:
    hint_data: list[dict] = load_json(hint_verification)
    keep_ids: Set[int] = extract_verification_only_ids(hint_data)

    completions: list[dict] = load_json(completions)
    comp_ids = {c["question_id"] for c in completions}
    keep_ids &= comp_ids

    keep_ids_sorted: List[int] = sorted(keep_ids)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(keep_ids_sorted, f, indent=2)

    print(f"Wrote {len(keep_ids_sorted)} IDs -> {output_file}")

    return output_file
