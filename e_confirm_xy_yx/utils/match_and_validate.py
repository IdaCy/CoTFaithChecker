#!/usr/bin/env python3
"""
Build a per-question, per-run comparison file for every A/B file-pair.

For every file “fileA*” in directory A the script finds the file with
the **same index** in directory B, plus a matcher file whose filename
*contains* the part of fileA’s name that appears **before** “_{key}_”.

It then creates JSON like

[
  {
    "question_id": 1,
    "question_yes_id": 98,
    "a_answers": ["YES", ...]          # 10 entries, run 0-9
    "b_answers": ["NO",  ...]          # 10 entries, run 0-9
    "same":      [False, ...]          # True = both YES or both NO
  },
  ...
]

and writes it to *directoryNew/fileA*.json (same basename as the
original A-file).

running:
python e_confirm_xy_yx/utils/match_and_validate.py \
    --dirA e_confirm_xy_yx/outputs/answers_gt/gt_NO_1 \
    --dirB e_confirm_xy_yx/outputs/answers_gt/gt_YES_1 \
    --matcher data/chainscope/questions_json/linked/gt_NO_1 \
    --key gt \
    --out e_confirm_xy_yx/outputs/matched_vals


Assumptions
-----------
* Every file in dirA has a *position-matched* file in dirB
  (i.e. alphabetical order 0..N matches).
* Every file contains exactly 10 runs per question, runs 0-9.
* Answer strings are exactly "YES" or "NO".
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import traceback


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_answers_per_question(file_path: Path) -> Dict[int, Dict[int, str]]:
    """Returns {question_id: {run: answer}}"""
    print(f"  ↳ loading answers from {file_path}")
    with file_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    by_q: Dict[int, Dict[int, str]] = {}
    for entry in data:
        qid = entry["question_id"]
        run = entry["run"]
        ans = entry["answer"].upper()
        by_q.setdefault(qid, {})[run] = ans
    print(f"    • loaded {len(by_q)} unique question_ids")
    return by_q


def answers_vector(qruns: Dict[int, str], runs: int = 10) -> List[str]:
    """Turns {run: answer} into a fixed-length list (runs 0-9)."""
    return [qruns.get(r) for r in range(runs)]


def bool_vector(a: List[str], b: List[str]) -> List[bool]:
    """True where both answers are equal & valid; False otherwise."""
    return [
        (aa == bb and aa in ("YES", "NO"))
        if aa is not None and bb is not None
        else False
        for aa, bb in zip(a, b)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main per-file routine
# ──────────────────────────────────────────────────────────────────────────────

def process_pair(
    fileA: Path,
    fileB: Path,
    matcher_dir: Path,
    key: str,
    dest_dir: Path,
):
    print(f"\n=== Processing pair ===")
    print(f"A: {fileA}")
    print(f"B: {fileB}")
    base = fileA.name

    # part before "_{key}_"
    if f"_{key}_" not in base:
        raise RuntimeError(f'Key "{key}" not in filename "{base}"')
    prefix = base.split(f"_{key}_", 1)[0]
    print(f"  prefix determined from filename: '{prefix}'")

    # locate matcher file
    matcher_files = sorted(matcher_dir.glob(f"*{prefix}*.json"))
    print(f"  matcher search pattern '*{prefix}*.json' "
          f"found {len(matcher_files)} file(s)")
    if not matcher_files:
        raise RuntimeError(f"No matcher file containing '{prefix}' in {matcher_dir}")
    matcher_path = matcher_files[0]
    print(f"  matcher file chosen: {matcher_path}")

    # load answer data
    a_by_q = load_answers_per_question(fileA)
    b_by_q = load_answers_per_question(fileB)

    # load matcher mapping
    print(f"  ↳ loading matcher mapping …")
    with matcher_path.open("r", encoding="utf-8") as fp:
        matcher = json.load(fp)
    mapping = {q["question_id"]: q["yes_question_id"] for q in matcher["questions"]}
    print(f"    • matcher contains {len(mapping)} mapped question_ids")

    # build output records
    out = []
    for idx, (qid, runs_dict) in enumerate(a_by_q.items(), 1):
        if qid not in mapping:
            print(f"    – skipping question_id {qid} (not in matcher)")
            continue

        yes_qid = mapping[qid]
        a_vec = answers_vector(runs_dict)
        b_vec = answers_vector(b_by_q.get(yes_qid, {}))
        same_vec = bool_vector(a_vec, b_vec)

        out.append(
            {
                "question_id": qid,
                "question_yes_id": yes_qid,
                "a_answers": a_vec,
                "b_answers": b_vec,
                "same": same_vec,
            }
        )

        if idx % 100 == 0:
            print(f"      processed {idx} questions so far …")

    print(f"  total questions emitted: {len(out)}")

    # write result
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / base
    with dest_path.open("w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print(f"✔ wrote {dest_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirA", required=True, type=Path)
    parser.add_argument("--dirB", required=True, type=Path)
    parser.add_argument("--matcher", required=True, type=Path)
    parser.add_argument("--key", required=True, help="key_string used in filenames")
    parser.add_argument("--out", required=True, type=Path, help="directoryNew")
    args = parser.parse_args()

    print(f"dirA     : {args.dirA}")
    print(f"dirB     : {args.dirB}")
    print(f"matcher  : {args.matcher}")
    print(f"key      : {args.key}")
    print(f"dest dir : {args.out}")

    filesA = sorted(args.dirA.glob("*.json"))
    filesB = sorted(args.dirB.glob("*.json"))

    print(f"\nFound {len(filesA)} *.json files in dirA "
          f"and {len(filesB)} in dirB")

    if len(filesA) != len(filesB):
        raise RuntimeError("dirA and dirB do not contain the same number of files")

    for i, (fa, fb) in enumerate(zip(filesA, filesB), 1):
        print(f"\n──────────────────────────────────────────────────────────")
        print(f"Pair {i}/{len(filesA)}")
        try:
            process_pair(
                fa,
                fb,
                matcher_dir=args.matcher,
                key=args.key,
                dest_dir=args.out,
            )
        except Exception as exc:
            print("\nERROR during this pair – traceback follows")
            traceback.print_exc()
            sys.exit(1)   # stop immediately so you know where it failed


if __name__ == "__main__":
    main()