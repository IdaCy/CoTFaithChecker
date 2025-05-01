# -------------------------------------------------
# embeddings_pipeline.py
# -------------------------------------------------
import json, os, time, logging
from typing import List, Dict, Optional, Sequence, Tuple

import torch
from accelerate.utils import gather_object
from sentence_transformers import SentenceTransformer  # local encoder
# If you prefer OpenAI embeddings comment the import above
# and uncomment the two lines below
# import openai
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- 1.  >>>>>>  your unchanged helpers  <<<<<< ------------------------------
from yourmodule.utils.prompt_constructor import construct_prompt     # noqa: F401
from yourmodule.capt5k.model_handler     import generate_completion  # noqa: F401
from yourmodule.helpers import (                                    # noqa: F401
    get_device, load_model_and_tokenizer, tokenize_instructions,
    StopOnThinkEnd
)

# ---------------------------------------------------------------------------
# 2.  *** NEW ***  universal embedding wrapper
# ---------------------------------------------------------------------------
class EmbeddingModel:
    """
    Handles *either* a local Sentence-Transformers encoder (default) *or*
    an OpenAI endpoint.  Switching is just a flag.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: torch.device | str | None = None,
        openai_api_key: str | None = None,
        remote: bool = False,
        batch_size: int = 64,
    ):
        self.remote = remote
        self.batch  = batch_size

        if remote:
            # --- OpenAI
            import openai  # local import avoids the dependency if unused
            openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if openai.api_key is None:
                raise ValueError("OPENAI_API_KEY env-var or argument required")
            self._model = "text-embedding-3-small" if model_name is None else model_name
        else:
            # --- local encoder
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(model_name, device=device)

    # ---------- public ------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Returns a *list* of float lists (so it’s JSON-serialisable).
        Keeps order = input order.
        """

        if self.remote:
            return self._encode_remote(texts)
        return [vec.tolist() for vec in self._model.encode(texts,
                                                           batch_size=self.batch,
                                                           convert_to_tensor=True,
                                                           normalize_embeddings=True)]

    # ---------- internal ----------------------------------------------------
    # If you want exponential back-off retry uncomment @retry decorator
    # @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _encode_remote(self, texts: Sequence[str]) -> List[List[float]]:
        import openai
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            rsp = openai.Embedding.create(model=self._model, input=chunk)
            # OpenAI guarantees order
            vectors.extend([d["embedding"] for d in rsp["data"]])
        return vectors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3.  *** NEW ***  main driver – mirrors generate_dataset_completions
# ---------------------------------------------------------------------------
def generate_dataset_embeddings(
    accelerator,
    model, tokenizer, model_name, device,
    dataset_name: str,
    hint_types: List[str],
    *,
    batch_size: int       = 8,
    max_new_tokens: int   = 512,
    n_questions: int | None = None,
    # ---- embedding specific
    embedder: EmbeddingModel | None = None,
    sentence_level: bool = True,
) -> None:
    """
    *Identical control flow* to your completion script, but writes embeddings.

    Each output record looks like
    {
      "question_id": 123,
      "hint_type":   "unhinted",
      "completion":  "... <think> ...",        # optional – keep if handy
      "sentence_embeddings": [ [..768..], ...] # list-of-lists
      "cot_embedding":       [..768..]         # only if sentence_level=False
    }
    """

    start = time.time()
    if accelerator.is_main_process:
        logging.info(f"Running on {accelerator.num_processes} GPU(s)")

    chat_template = get_chat_template(model_name)   # ← your helper

    # -------- prepare embedder once ----------------------------------------
    if embedder is None:
        embedder = EmbeddingModel()                 # defaults = MiniLM local

    for hint_type in hint_types:
        logging.info(f"--- Processing hint type: {hint_type} ---")

        q_path = os.path.join("data", dataset_name, "input_mcq_data.json")
        h_path = os.path.join("data", dataset_name, f"hints_{hint_type}.json")

        # ... unchanged ---
        data  = load_data(q_path)[:n_questions]
        hints = load_data(h_path)[:n_questions]

        rank, world = accelerator.process_index, accelerator.num_processes
        data  = data [rank::world]
        hints = hints[rank::world]

        # merge hints into questions exactly as before
        hint_dict = {h["question_id"]: h for h in hints}
        for entry in data:
            h = hint_dict.get(entry["question_id"])
            entry["hint_text"] = h.get("hint_text") if h else None

        # build prompts
        prompts = []
        for entry in data:
            prompt = construct_prompt(entry).rstrip() + "\n<think>"
            prompts.append({"question_id": entry["question_id"],
                            "prompt_text": prompt})

        # ------------------------------------------------------------------
        # (3)  generate CoT completions – RE-USE your existing function
        # ------------------------------------------------------------------
        cot_results = generate_completion(
            model, tokenizer, device, prompts,
            chat_template, batch_size, max_new_tokens
        )

        # ------------------------------------------------------------------
        # (4)  embed & package
        # ------------------------------------------------------------------
        packaged: List[Dict] = []

        for rec in cot_results:
            span = rec["completion"]                # already trimmed to <think>…​
            if sentence_level:
                sents  = _split_text_into_sentences(span)   # helper below
                vecs   = embedder.encode(sents)
                packaged.append({
                    "question_id": rec["question_id"],
                    "hint_type":   hint_type,
                    "sentence_embeddings": vecs,
                    # keep text if useful for debugging
                    # "sentences": sents
                })
            else:
                vec = embedder.encode([span])[0]
                packaged.append({
                    "question_id": rec["question_id"],
                    "hint_type":   hint_type,
                    "cot_embedding": vec,
                    # "completion": span
                })

        # gather & save (exactly as before)
        gathered = gather_object(packaged)
        if accelerator.is_main_process:
            merged = [item for sub in gathered for item in sub]
            _save_embeddings(merged, dataset_name, hint_type,
                             model_name, sentence_level, n_questions)

    if accelerator.is_main_process:
        logging.info(f"Total time: {time.time() - start:.2f} s")


# ---------------------------------------------------------------------------
# 4. helpers: split sentences + save
# ---------------------------------------------------------------------------
import re, json
def _split_text_into_sentences(txt: str) -> List[str]:
    """Tiny sentence splitter identical to the one in your annotator script."""
    txt = re.sub(r"\s+", " ", txt.strip())
    return [s.strip() for s in re.split(r"(?<=\.)\s+", txt) if s.strip()]

def _save_embeddings(data: List[Dict], dataset: str, hint: str,
                     model: str, sentence_level: bool, n_q: int | None):
    suffix = "sentence_embeds" if sentence_level else "cot_embeds"
    out = os.path.join("data", dataset, model, hint,
                       f"{suffix}_for_{n_q or 'all'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Embeddings saved to {out}")
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. convenience CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, accelerate

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_path", required=True,
                        help="path to your llama-8B (same as before)")
    parser.add_argument("--embedder_name", default=None,
                        help="Sentence-Transformers model OR OpenAI name")
    parser.add_argument("--openai", action="store_true",
                        help="Use OpenAI embeddings instead of local ST")
    parser.add_argument("--hint_types", nargs="+", default=["none", "verbalized", "unverbalized"])
    parser.add_argument("--n_questions", type=int)
    args = parser.parse_args()

    # accelerator & llama
    accelerator = accelerate.Accelerator()
    llama, tok, name, device = load_model_and_tokenizer(args.model_path)

    # embedder
    embedder = EmbeddingModel(model_name=args.embedder_name,
                              remote=args.openai)

    generate_dataset_embeddings(
        accelerator=accelerator,
        model=llama,
        tokenizer=tok,
        model_name=name,
        device=device,
        dataset_name=args.dataset,
        hint_types=args.hint_types,
        embedder=embedder,
        n_questions=args.n_questions,
    )
