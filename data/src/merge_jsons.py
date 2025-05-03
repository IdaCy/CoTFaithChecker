#!/usr/bin/env python3
"""
append_json.py
==============

Take two JSON files whose top-level element is a list and write an
output file that contains *file1* followed immediately by *file2*.

No deduplication, no key checks, no sorting—literally just list1 + list2.

Usage
-----
$ python append_json.py file1.json file2.json merged.json
python merge_jsons.py /root/CoTFaithChecker/data/mmlu/DeepSeek-R1-Distill-Llama-8B/none/completions_with_2001.json data/mmlu/DeepSeek-R1-Distill-Llama-8B/none/completions_with_1688.json data/mmlu/DeepSeek-R1-Distill-Llama-8B/none/completions_with_3689.json
"""

import argparse
import json
from pathlib import Path
import sys


def load_list(path: Path):
    """Load a JSON file and confirm the top level is a list."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] {path} is not valid JSON: {e}")

    if not isinstance(data, list):
        sys.exit(f"[ERROR] {path} does not contain a JSON *array* at the top level.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append the contents of two JSON arrays."
    )
    parser.add_argument("file1", help="First JSON file (kept in front)")
    parser.add_argument("file2", help="Second JSON file (appended)")
    parser.add_argument("outfile", help="Where to write the merged JSON")
    args = parser.parse_args()

    p1, p2, out = map(Path, (args.file1, args.file2, args.outfile))

    list1 = load_list(p1)
    list2 = load_list(p2)

    merged = list1 + list2

    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("----------------------------------------------------")
    print(f"{p1.name:20}: {len(list1):>6} items")
    print(f"{p2.name:20}: {len(list2):>6} items")
    print(f"→ written to {out} : {len(merged):>6} items total")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
