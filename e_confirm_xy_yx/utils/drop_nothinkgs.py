#!/usr/bin/env python3
"""
Filter LLM completions.

•  Keeps only objects whose "completion" field
   – contains a "</think>" marker, **and**
   – has at least one accepted answer token *after* that marker.

•  Writes the filtered list to an output JSON.

•  Also writes an extra JSON containing the dropped question_id values.
   The file is named <input-basename>_dropped_ids.json.

Can be run either

A.  On a single pair of input/output files
    python filter_completions.py input.json output.json

B.  On a directory tree (only top-level *.json files are processed)
    python filter_completions.py --in_dir data_in --out_dir data_out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

ACCEPTED_TOKENS: List[str] = [
    "YES", "NO",
    " Yes ", " No ",
    " Yes.", " No.",
    " Yes,", "No,",
    "\nYes,", "\nNo,",
    "\nYes.", "\nNo.",
    "\nYes ", "\nNo ",
]

THINK_MARKER = "</think>"


def passes_filters(completion: str) -> bool:
    """Return True if the completion should be kept."""
    if THINK_MARKER not in completion:
        return False

    post_think = completion.split(THINK_MARKER, 1)[1]  # text *after* </think>
    return any(token in post_think for token in ACCEPTED_TOKENS)


def process_file(in_path: Path, out_path: Path) -> None:
    """Filter one JSON file and write outputs."""
    try:
        data: List[Dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        sys.exit(f"✖ Failed to read {in_path}: {exc}")

    kept: List[Dict[str, Any]] = []
    dropped_ids: List[Any] = []

    for obj in data:
        completion = obj.get("completion", "")
        if passes_filters(completion):
            kept.append(obj)
        else:
            dropped_ids.append(obj.get("question_id"))

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    dropped_path = out_path.with_name(f"{in_path.stem}_dropped_ids.json")
    dropped_path.write_text(json.dumps(dropped_ids, indent=2), encoding="utf-8")

    print(
        f"• {in_path.name}: kept {len(kept):>4}, dropped {len(dropped_ids):>4} "
        f"→ {out_path.relative_to(out_path.parent)}  /  {dropped_path.relative_to(dropped_path.parent)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter LLM completion JSON files.", formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "input",
        nargs="?",
        help="Input JSON file (file mode)",
    )
    group.add_argument(
        "--in_dir",
        help="Directory containing JSON files to process (directory mode)",
    )

    parser.add_argument(
        "output",
        nargs="?",
        help="Output JSON file (file mode)",
    )
    parser.add_argument(
        "--out_dir",
        help="Directory where processed JSON files will be written (directory mode)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:  # ─── File mode ───────────────────────────────────────────────
        if not args.output:
            sys.exit("✖ In file mode you must supply both input and output file names.")
        in_path = Path(args.input)
        out_path = Path(args.output)
        process_file(in_path, out_path)

    else:  # ─── Directory mode ──────────────────────────────────────────────────
        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir or in_dir.parent / f"{in_dir.name}_filtered")
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(p for p in in_dir.glob("*.json") if p.is_file())
        if not json_files:
            sys.exit(f"✖ No JSON files found in {in_dir}")

        for in_path in json_files:
            out_path = out_dir / in_path.name
            process_file(in_path, out_path)


if __name__ == "__main__":
    main()
