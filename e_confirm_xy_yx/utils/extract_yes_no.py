"""
Usage:
    python extract_yes_no.py path/to/input_dir path/to/output_dir

python e_confirm_xy_yx/utils/extract_yes_no.py \
    e_confirm_xy_yx/outputs/lt_NO \
    e_confirm_xy_yx/outputs/answers_lt_NO
"""

import argparse
import json
import pathlib
import re
import sys

_YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def final_yes_no(text: str) -> str | None:
    """
    Return the last explicit YES or NO found in *text* (case-insensitive).
    If neither is present, return None.
    """
    matches = _YES_NO_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def process_file(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Read *in_path*, extract answers, and write the slim file to *out_path*."""
    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"⚠️  Skipping {in_path} (invalid JSON): {exc}", file=sys.stderr)
        return

    results: list[dict] = []
    for entry in data:
        answer = final_yes_no(entry.get("completion", ""))
        if answer is None:
            # If you’d rather include unanswered items,
            # change this to `answer = ""` instead of continue.
            continue

        results.append(
            {
                "question_id": entry.get("question_id"),
                "run": entry.get("run"),
                "answer": answer,
            }
        )

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# -------- command-line interface --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract_yes_no",
        description=(
            "Scan each JSON file in INPUT_DIR, extract the final YES/NO from "
            "every completion, and write a parallel file to OUTPUT_DIR."
        ),
    )
    parser.add_argument("input_dir", help="directory containing source JSON files")
    parser.add_argument("output_dir", help="directory to receive the stripped JSON files")
    args = parser.parse_args()

    in_dir = pathlib.Path(args.input_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()

    if not in_dir.is_dir():
        parser.error(f"input_dir {in_dir} is not a directory")

    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in sorted(in_dir.glob("*.json")):
        out_path = out_dir / in_path.name
        process_file(in_path, out_path)


if __name__ == "__main__":
    main()
