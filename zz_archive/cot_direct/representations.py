
"""Extract averaged hidden‑state vectors for segments using 🤗 Transformers."""
from pathlib import Path
from typing import List, Optional
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel

__all__ = ["RepresentationExtractor"]

class RepresentationExtractor:
    def __init__(self, model_name: str = "gpt2", cache_dir: str = ".cache/embeddings", device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache(self, text: str, layer: int):
        fname = f"{abs(hash(text))}_L{layer}.npy"
        return self.cache_dir / fname

    def embed_segment(self, text: str, layer: int = -1):
        cache_file = self._cache(text, layer)
        if cache_file.exists():
            return np.load(cache_file)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        vec = out.hidden_states[layer][0].mean(dim=0).cpu().numpy()
        np.save(cache_file, vec)
        return vec

    def bulk_embed(self, texts: List[str], layer: int = -1):
        return np.stack([self.embed_segment(t, layer=layer) for t in texts])
