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
# debug helper
# ──────────────────────────────────────────────────────────────────────────────
def make_debug(enabled: bool):
    def _dbg(*args, **kwargs):
        if enabled:
            print(*args, **kwargs)
    return _dbg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_answers_per_question(file_path: Path, dbg) -> Dict[int, Dict[int, str]]:
    """Returns {question_id: {run: answer}}"""
    dbg(f"  ↳ loading answers from {file_path}")
    with file_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    by_q: Dict[int, Dict[int, str]] = {}
    for entry in data:
        qid = entry["question_id"]
        run = entry["run"]
        ans = entry["answer"].upper()
        by_q.setdefault(qid, {})[run] = ans
    dbg(f"    • loaded {len(by_q)} unique question_ids")
    return by_q


def answers_vector(qruns: Dict[int, str], runs: int = 10) -> List[str]:
    """Turns {run: answer} into a fixed-length list (runs 0-9)."""
    return [qruns.get(r) for r in range(runs)]


def bool_vector(a: List[str], b: List[str]) -> List[bool | None]:
    """
    • True   → both answers are 'YES' or both 'NO'
    • False  → one 'YES' and the other 'NO'
    • None   → either answer is missing (None)
    """
    result: List[bool | None] = []
    for aa, bb in zip(a, b):
        if aa is None or bb is None:
            result.append(None)              # propagate missing values
        else:
            result.append(aa == bb)          # True if identical, else False
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main per-file routine
# ──────────────────────────────────────────────────────────────────────────────
def process_pair(
    fileA: Path,
    fileB: Path,
    matcher_dir: Path,
    key: str,
    dest_dir: Path,
    dbg,
):
    dbg(f"\n=== Processing pair ===\nA: {fileA}\nB: {fileB}")
    base = fileA.name

    # split filename at _{key}_
    if f"_{key}_" not in base:
        raise RuntimeError(f'Key "{key}" not in filename "{base}"')
    prefix = base.split(f"_{key}_", 1)[0]
    dbg(f"  prefix determined from filename: '{prefix}'")

    # matcher file
    matcher_files = sorted(matcher_dir.glob(f"*{prefix}*.json"))
    dbg(f"  matcher search '*{prefix}*.json' found {len(matcher_files)} file(s)")
    if not matcher_files:
        raise RuntimeError(f"No matcher file containing '{prefix}' in {matcher_dir}")
    matcher_path = matcher_files[0]
    dbg(f"  matcher file chosen: {matcher_path}")

    # load answers
    a_by_q = load_answers_per_question(fileA, dbg)
    b_by_q = load_answers_per_question(fileB, dbg)

    # load mapping
    dbg("  ↳ loading matcher mapping …")
    with matcher_path.open("r", encoding="utf-8") as fp:
        matcher = json.load(fp)
    mapping = {q["question_id"]: q["yes_question_id"] for q in matcher["questions"]}
    dbg(f"    • matcher contains {len(mapping)} mapped question_ids")

    # build output
    out = []
    for qid, runs_dict in a_by_q.items():
        if qid not in mapping:
            dbg(f"    – skipping question_id {qid} (not in matcher)")
            continue

        yes_qid = mapping[qid]
        a_vec = answers_vector(runs_dict)
        b_vec = answers_vector(b_by_q.get(yes_qid, {}))

        # ── diagnostics for missing runs ────────────────────────────────────
        missing_a = [idx for idx, val in enumerate(a_vec) if val is None]
        missing_b = [idx for idx, val in enumerate(b_vec) if val is None]
        if missing_a or missing_b:
            print(
                f"[MISSING] {base}  "
                f"qid {qid} → {yes_qid}  "
                f"A missing runs {missing_a or '—'}  "
                f"B missing runs {missing_b or '—'}"
            )

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

    dbg(f"  total questions emitted: {len(out)}")

    # write result
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / base
    with dest_path.open("w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    dbg(f"✔ wrote {dest_path}")


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
    parser.add_argument("--debug", action="store_true",
                        help="show verbose progress logs")
    args = parser.parse_args()

    dbg = make_debug(args.debug)

    dbg(f"dirA: {args.dirA}\ndirB: {args.dirB}\nmatcher: {args.matcher}")
    dbg(f"key: {args.key}\ndest dir: {args.out}")

    filesA = sorted(args.dirA.glob("*.json"))
    filesB = sorted(args.dirB.glob("*.json"))

    dbg(f"\nFound {len(filesA)} *.json files in dirA "
        f"and {len(filesB)} in dirB")

    if len(filesA) != len(filesB):
        raise RuntimeError("dirA and dirB do not contain the same number of files")

    for i, (fa, fb) in enumerate(zip(filesA, filesB), 1):
        dbg(f"\n──────────────────────────────────────────────────────────\nPair {i}/{len(filesA)}")
        try:
            process_pair(
                fa, fb,
                matcher_dir=args.matcher,
                key=args.key,
                dest_dir=args.out,
                dbg=dbg,
            )
        except Exception:
            print("\nERROR during this pair – traceback follows")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
