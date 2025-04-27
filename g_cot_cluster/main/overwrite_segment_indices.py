#!/usr/bin/env python3
"""
A script to recalculate the `start` and `end` character indices in a
`segmented_completions` JSON file based on the actual positions of each
segment within the full completion text found in a corresponding
`completions` JSON file.

The algorithm follows the steps described by the user:
1. Load both JSON files (they must contain lists of objects keyed by
   `question_id`).
2. For each shared `question_id`, walk through the segments in order,
   locating each segment's text inside the full completion string. Only
   the first/last eight characters of each segment are used for the
   search (or the full text if it is shorter than eight characters).
3. Replace the original `start`/`end` values with the ones discovered at
   runtime.
4. Write the updated `segmented_completions` structure back to disk.
5. Log any discrepancies (e.g., when a prefix or suffix cannot be
   located) together with the current `question_id`.

Example
=======
    python overwrite_segment_indices.py completions.json segmented_completions.json \
        -o segmented_completions_updated.json

If the first two `question_id`s do not match between the files, the
script emits a warning but will still attempt to process all IDs that
appear in *both* files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence


###############################################################################
# Helpers
###############################################################################

def load_json(path: Path | str) -> Any:
    """Read a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Any, path: Path | str) -> None:
    """Write *data* to *path* as UTF‑8 JSON with nice indentation."""
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def _find(haystack: str, needle: str, start: int) -> int:
    """Return the first index of *needle* in *haystack* at or after *start*.

    If *needle* is not found, ``-1`` is returned (to mirror ``str.find``).
    """
    return haystack.find(needle, start)


###############################################################################
# Core logic per question_id
###############################################################################

def _process_single_question(
    qid: int,
    full_completion: str,
    segments: Sequence[Dict[str, Any]],
) -> bool:
    """Update *segments* in‑place; return ``True`` on full success."""

    cursor = 0  # Where to start searching in *full_completion* next
    success = True

    for seg in segments:
        text = seg.get("text", "")
        if not text:
            logging.error("[QID %s] Empty segment text encountered; skipping.", qid)
            success = False
            continue

        # Steps 4 & 6: prefixes/suffixes (max 8 chars, or entire text if shorter)
        prefix = text[:8] if len(text) >= 8 else text
        suffix = text[-8:] if len(text) >= 8 else text

        # Step 5: locate prefix
        start_idx = _find(full_completion, prefix, cursor)
        if start_idx == -1:
            logging.error(
                "[QID %s] Prefix '%s' not found in completion at or after index %s",
                qid,
                prefix,
                cursor,
            )
            success = False
            continue  # Cannot proceed with this segment

        seg["start"] = start_idx

        # Step 7: locate suffix, starting from start_idx
        end_idx = _find(full_completion, suffix, start_idx)
        if end_idx == -1:
            logging.error(
                "[QID %s] Suffix '%s' not found in completion after index %s",
                qid,
                suffix,
                start_idx,
            )
            success = False
            continue

        seg["end"] = end_idx + len(suffix) - 1  # Inclusive index, per spec

        # Step 8: move cursor to char right after this segment
        cursor = seg["end"] + 1

    return success


###############################################################################
# Main entry‑point
###############################################################################

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute start/end indices inside a segmented_completions JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "completions",
        type=Path,
        help="Path to the completions JSON file",
    )
    parser.add_argument(
        "segmented_completions",
        type=Path,
        help="Path to the segmented_completions JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("segmented_completions_updated.json"),
        help="Where to write the updated segmented_completions file",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.loglevel))

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    completions_list: List[Dict[str, Any]] = load_json(args.completions)
    segments_list: List[Dict[str, Any]] = load_json(args.segmented_completions)

    # Index by question_id for fast lookup
    completions_by_id: Dict[int, Dict[str, Any]] = {
        entry["question_id"]: entry for entry in completions_list
    }
    segments_by_id: Dict[int, Dict[str, Any]] = {
        entry["question_id"]: entry for entry in segments_list
    }

    # Step 2: Check first two IDs match
    first_two_comp = [entry["question_id"] for entry in completions_list[:2]]
    first_two_segm = [entry["question_id"] for entry in segments_list[:2]]
    if set(first_two_comp) != set(first_two_segm):
        logging.warning(
            "First two question_ids differ between files: %s vs %s",
            first_two_comp,
            first_two_segm,
        )

    # Iterate through the intersection of IDs (order ascending)
    common_ids = sorted(completions_by_id.keys() & segments_by_id.keys())
    if not common_ids:
        logging.error("No overlapping question_ids found between the supplied files.")
        raise SystemExit(1)

    overall_success = True
    for qid in common_ids:
        comp_entry = completions_by_id[qid]
        seg_entry = segments_by_id[qid]

        ok = _process_single_question(
            qid=qid,
            full_completion=comp_entry["completion"],
            segments=seg_entry["segments"],
        )
        overall_success = overall_success and ok

    # Persist updated segments JSON
    save_json(list(segments_by_id.values()), args.output)

    if overall_success:
        logging.info("All segments processed successfully. Output written to %s", args.output)
    else:
        logging.warning(
            "Finished with some errors. See log for details. Output written to %s",
            args.output,
        )


if __name__ == "__main__":
    main()
