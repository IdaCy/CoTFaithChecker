#!/usr/bin/env python3
"""
    python data/src/json_merge.py fileA.json fileB.json merged.json
"""

import json
import sys
from pathlib import Path

def merge_json_arrays(file_a: Path, file_b: Path, outfile: Path) -> None:
    text_a = file_a.read_text(encoding="utf-8").rstrip()
    text_b = file_b.read_text(encoding="utf-8").lstrip()

    if not text_a.endswith("]"):
        raise ValueError(f"{file_a} doesn’t end with a closing bracket ]")
    if not text_b.startswith("["):
        raise ValueError(f"{file_b} doesn’t start with an opening bracket [")

    text_a = text_a[:-1]                     # drop trailing ]
    text_b = text_b[1:]                      # drop leading  [

    if text_a and text_a[-1] not in "[,":
        text_a += ","

    merged_text = f"[{text_a}{text_b}"      # text_b already ends with ]

    merged_obj = json.loads(merged_text)     # raises if invalid
    outfile.write_text(
        json.dumps(merged_obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_json_arrays.py fileA.json fileB.json merged.json")
        sys.exit(1)

    merge_json_arrays(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
