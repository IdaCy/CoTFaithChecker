import json, datetime, logging, pickle, os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler

from h_hidden_space.capture_f1.model_handler import load_model_and_tokenizer
from h_hidden_space.capture_f1.pipeline       import get_chat_template
from a_confirm_posthoc.utils.prompt_constructor import construct_prompt


def register_steering_hook(model, layer_idx: int,
                           direction: torch.Tensor, alpha: float):
    direction = direction.unsqueeze(0).unsqueeze(0).to(model.device)
    direction_scaled = alpha * direction

    def _hook(module, _, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest   = output[1:]
        else:
            hidden = output
            rest   = None

        steer = direction_scaled.to(hidden.dtype)   # cast to bf16 / fp16 / fp32
        hidden = hidden + steer

        return (hidden, *rest) if rest is not None else hidden

    return model.model.layers[layer_idx].register_forward_hook(_hook)



def run_steered_generation(
        model_path: str,
        probe_path: str,
        *,
        prompts: List[str] = None,
        dataset_name: Optional[str] = None,
        hint_type: Optional[str] = None,
        n_questions: Optional[int] = None,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        alpha: float = 5.0,
        scaler_path: Optional[str] = None,
        output_jsonl: Optional[str] = None,
):
    assert (prompts is not None) ^ (dataset_name is not None), \
        "Give either prompts OR (dataset_name & friends)"

    probe = pickle.load(open(probe_path, "rb"))
    layer_idx  = probe["layer"]
    delta_mu   = torch.tensor(probe["delta_mu"])         # (hidden,)

    if scaler_path is not None:
        scaler: StandardScaler = pickle.load(open(scaler_path, "rb"))
        # delta*myu was saved in standardised space -> conv back to raw space
        delta_mu = delta_mu * torch.tensor(scaler.scale_)  # undo /std
        # (no +mean term because delte*myu is a difference)

    model, tokenizer, model_name, device = load_model_and_tokenizer(model_path)
    chat_template = get_chat_template(model_name)

    if prompts is None:
        questions_fp = Path("data") / dataset_name / "input_mcq_data.json"
        with open(questions_fp) as f:
            data = json.load(f)[:n_questions]

        prompts = [construct_prompt(e) for e in data]

    hook_handle = register_steering_hook(model, layer_idx, delta_mu, alpha)

    results = []
    model.eval()
    torch.manual_seed(0)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(
            [chat_template.format(instruction=p) for p in batch_prompts],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )

        # strip prompt part off to -> pure completion
        prompt_lens = enc.input_ids.shape[1]
        completions = tokenizer.batch_decode(
            gen.sequences[:, prompt_lens:], skip_special_tokens=True
        )

        for p, c in zip(batch_prompts, completions):
            results.append({"prompt": p, "completion": c})

    hook_handle.remove()

    if output_jsonl:
        with open(output_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    return results


if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser(
        description="Steer DeepSeek-R1-Distill-Llama-8B with a Δμ probe",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--probe_path", required=True)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_questions", type=int, default=100,
                    help="If using dataset prompts, how many questions.")
    ap.add_argument("--dataset_name", default="mmlu")
    ap.add_argument("--hint_type", default="none")
    ap.add_argument("--output_jsonl",
                    help="Write results to this file if given")
    args = ap.parse_args()

    res = run_steered_generation(
        model_path=args.model_path,
        probe_path=args.probe_path,
        dataset_name=args.dataset_name,
        hint_type=args.hint_type,
        n_questions=args.n_questions,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        alpha=args.alpha,
        output_jsonl=args.output_jsonl,
    )
    print(f"Done – generated {len(res)} samples.")
