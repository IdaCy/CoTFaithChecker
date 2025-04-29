#!/usr/bin/env python3
"""
link_yes_no.py

Cross‑link every pair of files in two *input* directories that were produced by
`convert_questions.py` – one directory whose questions were answered 'NO', the
other 'YES'.

The script now writes the linked copies to two *output* directories instead of
modifying the originals.  One output directory will receive the updated NO
files, the other the updated YES files.

For each matching question whose x/y values are reversed between a NO file and
its corresponding YES file, add:

    • yes_question_id   (to the NO question object)
    • no_question_id    (to the YES question object)

Usage
-----
    python link_yes_no.py IN_NO_DIR IN_YES_DIR OUT_NO_DIR OUT_YES_DIR
    python link_yes_no.py IN_NO_DIR IN_YES_DIR OUT_NO_DIR OUT_YES_DIR -s _linked

python e_confirm_xy_yx/utils/link_yes_no.py \
    data/chainscope/questions_json/unlinked/gt_NO_12 \
    data/chainscope/questions_json/unlinked/gt_YES_12 \
    data/chainscope/questions_json/linked/gt_NO_1 \
    data/chainscope/questions_json/linked/gt_YES_1

With the ``-s / --suffix`` option each output file name will be suffixed before
its extension (e.g. ``foo.json`` → ``foo_linked.json``).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple


def load_json(path: Path) -> Dict:
    """Read *path* and return the decoded JSON dict."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path) -> None:
    """Write *data* as pretty‑printed JSON to *path*."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cross_link(no_path: Path, yes_path: Path) -> None:
    """Add reciprocal IDs between the questions contained in *no_path* and *yes_path*."""
    no_data = load_json(no_path)
    yes_data = load_json(yes_path)

    # Build lookup: (x, y) ➜ YES‑question dict
    yes_lookup: Dict[Tuple[str, str], Dict] = {
        (q["x_value"], q["y_value"]): q for q in yes_data["questions"]
    }

    for q_no in no_data["questions"]:
        reversed_pair = (q_no["y_value"], q_no["x_value"])
        q_yes = yes_lookup.get(reversed_pair)
        if q_yes:
            q_no["yes_question_id"] = q_yes["question_id"]
            q_yes["no_question_id"] = q_no["question_id"]

    save_json(no_data, no_path)
    save_json(yes_data, yes_path)
    print(f"✓ linked {no_path.name} ↔ {yes_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross‑link matching NO/YES question files and write the results "
            "to separate output directories."
        )
    )
    parser.add_argument("in_no_dir", help="Input directory with *_NO_* JSON files")
    parser.add_argument("in_yes_dir", help="Input directory with *_YES_* JSON files")
    parser.add_argument("out_no_dir", help="Output directory for linked NO files")
    parser.add_argument("out_yes_dir", help="Output directory for linked YES files")
    parser.add_argument(
        "-s",
        "--suffix",
        help=(
            "Suffix to append to every output filename before the extension "
            "(e.g. '_linked' → foo.json → foo_linked.json)."
        ),
        default="",
    )
    args = parser.parse_args()

    # ---- Validate directories ------------------------------------------------
    in_no_dir = Path(args.in_no_dir)
    in_yes_dir = Path(args.in_yes_dir)
    out_no_dir = Path(args.out_no_dir)
    out_yes_dir = Path(args.out_yes_dir)

    for d, label in [
        (in_no_dir, "input NO"),
        (in_yes_dir, "input YES"),
    ]:
        if not d.is_dir():
            parser.error(f"{label.capitalize()} directory '{d}' does not exist or is not a directory.")

    # Create output dirs up‑front so later Path operations succeed
    out_no_dir.mkdir(parents=True, exist_ok=True)
    out_yes_dir.mkdir(parents=True, exist_ok=True)

    # ---- Gather files --------------------------------------------------------
    no_files = sorted(in_no_dir.glob("*.json"))
    yes_files = sorted(in_yes_dir.glob("*.json"))

    if len(no_files) != len(yes_files):
        print(
            f"⚠ Warning: directory sizes differ "
            f"({len(no_files)} NO files vs {len(yes_files)} YES files). "
            f"Only matching indices will be processed.",
            file=sys.stderr,
        )

    # ---- Process pairs -------------------------------------------------------
    for no_f, yes_f in zip(no_files, yes_files):
        # Build target paths inside the *output* directories
        no_target = out_no_dir / (
            (no_f.stem + args.suffix) + no_f.suffix
        )
        yes_target = out_yes_dir / (
            (yes_f.stem + args.suffix) + yes_f.suffix
        )

        # Copy originals to the output location (overwriting if they already exist)
        shutil.copyfile(no_f, no_target)
        shutil.copyfile(yes_f, yes_target)

        # Cross‑link the freshly copied files in place
        try:
            cross_link(no_target, yes_target)
        except Exception as e:
            print(
                f"✗ failed linking {no_f.name} ↔ {yes_f.name}: {e}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
