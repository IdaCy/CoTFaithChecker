import json
from pathlib import Path
from typing import Dict, List, Optional
import re
from collections import defaultdict
from e_confirm_xy_yx.main.data_loader import detect_cluster

import torch
from e_confirm_xy_yx.main.logging_utils import init_logger
from e_confirm_xy_yx.main.data_loader import load_dataset
from e_confirm_xy_yx.main.prompt_builder import PromptBuilder

__all__ = ["run_inference"]

logger = init_logger(Path.cwd() / "logs", "inference")


def _save_hidden_or_attention(tensor_list: List[torch.Tensor], out_file: Path):
    """
    Utility for serialising tensors.
    """
    torch.save(tensor_list, out_file)
    logger.info(f"Saved tensor list → {out_file}")


def _generate(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: Optional[int],
    save_hidden: bool,
    hidden_layers: List[int],
    save_attention: bool,
    attn_layers: List[int],
    temperature: float,
    top_p: float,
) -> Dict:
    """
    Batched generation with nucleus sampling.
    Generates *one* completion per prompt. Sampling parameters are now
    explicit so the caller can produce multiple reasoning chains.
    """
    encodings = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = encodings.input_ids.shape[1]

    gen_kwargs = {
        "output_hidden_states":     save_hidden,
        "output_attentions":        save_attention,
        "return_dict_in_generate":  True,
        "max_new_tokens":           1024 if max_new_tokens is None else max_new_tokens,
        # NEW sampling controls ↓↓↓
        "do_sample":   True,
        "temperature": temperature,
        "top_p":       top_p,
    }

    with torch.no_grad():
        outputs = model.generate(**encodings, **gen_kwargs)

    completions = tokenizer.batch_decode(
        outputs.sequences[:, prompt_len:], skip_special_tokens=True
    )

    extra: Dict[str, List] = {}
    if save_hidden:
        hidden = [
            [outputs.hidden_states[i][layer].cpu() for layer in hidden_layers]
            for i in range(len(prompts))
        ]
        extra["hidden"] = hidden
    if save_attention:
        atts = [
            [outputs.attentions[i][layer].cpu() for layer in attn_layers]
            for i in range(len(prompts))
        ]
        extra["attention"] = atts

    return {"completions": completions, **extra}


def _process_single_file(
    json_path: Path,
    prompt_builder: PromptBuilder,
    model,
    tokenizer,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_new_tokens: Optional[int],
    save_hidden: bool,
    hidden_layers: List[int],
    save_attention: bool,
    attn_layers: List[int],
    output_dir: Path,
    n_runs: int,
    temperature: float,
    top_p: float,
) -> Path:
    """
    Run `n_runs` independent samplings per question.
    Every output record now carries the `run` index (0-based).
    """
    logger.info(f"Processing {json_path.name}")
    data       = load_dataset(json_path)
    questions  = data["questions"]
    prompts    = prompt_builder.batch([q["q_str"] for q in questions])
    records: List[Dict] = []

    for run_idx in range(n_runs):
        logger.debug(f"  ↳ run {run_idx + 1}/{n_runs}")
        completions: List[str] = []

        # batched sampling for this run
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            gen_out = _generate(
                model,
                tokenizer,
                batch_prompts,
                device,
                max_new_tokens,
                save_hidden,
                hidden_layers,
                save_attention,
                attn_layers,
                temperature,
                top_p,
            )
            completions.extend(gen_out["completions"])

            # optional tensor dumps – include run-index in filename so they
            # don’t overwrite each other
            if save_hidden:
                _save_hidden_or_attention(
                    gen_out["hidden"],
                    output_dir / f"{json_path.stem}_run{run_idx}_hidden.pt",
                )
            if save_attention:
                _save_hidden_or_attention(
                    gen_out["attention"],
                    output_dir / f"{json_path.stem}_run{run_idx}_attn.pt",
                )

        # keep bookkeeping for every completion
        for q, comp in zip(questions, completions):
            row = {
                "question_id": q["question_id"],
                "run":         run_idx,
                "completion":  comp,
            }
            if "yes_question_id" in q:
                row["yes_question_id"] = q["yes_question_id"]
            if "no_question_id" in q:
                row["no_question_id"] = q["no_question_id"]
            records.append(row)

    out_path = output_dir / f"{json_path.stem}_{model_name}_completions.json"
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved {len(records)} completions → {out_path}")
    return out_path


def run_inference(
    dataset_files: List[Path],
    prompt_builder: PromptBuilder,
    model,
    tokenizer,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_new_tokens: Optional[int],
    save_hidden: bool,
    hidden_layers: List[int],
    save_attention: bool,
    attn_layers: List[int],
    output_dir: Path,
    n_runs: int,
    temperature: float,
    top_p: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: defaultdict[tuple, list] = defaultdict(list)
    pattern = re.compile(r"_(gt|lt)_(YES|NO)_")

    for fp in dataset_files:
        out_path = _process_single_file(
            fp,
            prompt_builder,
            model,
            tokenizer,
            model_name,
            device,
            batch_size,
            max_new_tokens,
            save_hidden,
            hidden_layers,
            save_attention,
            attn_layers,
            output_dir,
            n_runs,
            temperature,
            top_p,
        )

        # aggregation unchanged …
        if match := pattern.search(fp.stem):
            comparison, expected = match.groups()
        else:
            comparison, expected = "unknown", "unknown"

        cluster = detect_cluster(fp.stem)
        with out_path.open() as f:
            buckets[(cluster, comparison, expected)].extend(json.load(f))

    # … remainder of function unchanged …
    cluster_dir = output_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for (cluster, cmp, exp), recs in buckets.items():
        fname = f"{cluster}_{cmp}_{exp}_{model_name}_completions.json"
        with (cluster_dir / fname).open("w") as f:
            json.dump(recs, f, indent=2)
        logger.info(f"Aggregated {len(recs)} records → {cluster_dir / fname}")
