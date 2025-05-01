#!/usr/bin/env python3
"""

    python create_segment_indices.py completions.json segmented_completions.json \
        -o segmented_completions_updated.json

    python g_cot_cluster/main/create_segment_indices.py \
        c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B/completions/none.json \ 
        c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B/3phrase_splitting_none_filtered.json \
        -o c_cluster_analysis/outputs/hints/mmlu/DeepSeek-R1-Distill-Llama-8B/3phrase_splitting_none_filtered_updated.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Any, path: Path | str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def _find(haystack: str, needle: str, start: int) -> int:
    return haystack.find(needle, start)


def _segment_text(seg: Dict[str, Any]) -> str:
    return seg.get("segment", seg.get("text", ""))

def _process_single_question(
    qid: int,
    full_completion: str,
    segments: Sequence[Dict[str, Any]],
) -> bool:
    cursor = 0
    success = True

    for seg in segments:
        text = _segment_text(seg)
        if not text:
            logging.error("[QID %s] Empty segment text; skipping.", qid)
            success = False
            continue

        prefix = text[:8] if len(text) >= 8 else text
        suffix = text[-8:] if len(text) >= 8 else text

        start_idx = _find(full_completion, prefix, cursor)
        if start_idx == -1 or abs(start_idx - cursor) > 10:
            win_lo = max(0, cursor - 10)
            win_hi = min(len(full_completion), cursor + 10 + len(prefix))
            local = full_completion[win_lo:win_hi].find(prefix)
            start_idx = win_lo + local if local != -1 else cursor
        seg["start"] = start_idx
        end_idx = _find(full_completion, suffix, start_idx)
        if end_idx != -1:
            seg["end"] = end_idx + len(suffix) - 1        # normal path
        else:
            max_len = min(len(text), len(full_completion) - start_idx)
            matched = 0
            for i in range(max_len):
                if full_completion[start_idx + i] == text[i]:
                    matched = i + 1
                else:
                    break
            if matched:
                seg["end"] = start_idx + matched - 1
                logging.debug("[QID %s] Suffix not found; used fuzzy end=%s",
                              qid, seg["end"])
            else:
                logging.error(
                    "[QID %s] Could not determine end index for segment.", qid
                )
                success = False
                continue

        cursor = seg["end"] + 1

    return success


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recompute start/end indices inside a segmented_completions JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("completions", type=Path,
                        help="Path to the completions JSON file")
    parser.add_argument("segmented_completions", type=Path,
                        help="Path to the segmented_completions JSON file")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("segmented_completions_updated.json"),
                        help="Where to write the updated file")
    parser.add_argument("--loglevel", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity")

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.loglevel))

    completions_list: List[Dict[str, Any]] = load_json(args.completions)
    segments_list:    List[Dict[str, Any]] = load_json(args.segmented_completions)

    completions_by_id = {e["question_id"]: e for e in completions_list}
    segments_by_id    = {e["question_id"]: e for e in segments_list}

    first_two_comp = [e["question_id"] for e in completions_list[:2]]
    first_two_segm = [e["question_id"] for e in segments_list[:2]]
    if set(first_two_comp) != set(first_two_segm):
        logging.warning(
            "First two question_ids differ between files: %s vs %s",
            first_two_comp, first_two_segm,
        )

    common_ids = sorted(completions_by_id.keys() & segments_by_id.keys())
    if not common_ids:
        logging.error("No overlapping question_ids found between the supplied files.")
        raise SystemExit(1)

    overall_success = True
    for qid in common_ids:
        comp_entry = completions_by_id[qid]
        seg_entry  = segments_by_id[qid]

        ok = _process_single_question(
            qid,
            full_completion=comp_entry["completion"],
            segments=seg_entry.get("annotations", seg_entry.get("segments", [])),
        )
        overall_success = overall_success and ok

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
