"""Text embeddings for meaning-based lookup (paper: EmbeddingGemma on CPU).

All vectors are L2-normalised, so cosine similarity is a dot product. A flat
numpy matrix search is mathematically the same as the paper's FAISS flat index.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from . import config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Embedder:
    def __init__(self) -> None:
        self._model = None
        self.name: str | None = None
        self._prompt_name: str | None = None
        self._memo: dict[str, np.ndarray] = {}

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
            self.name = config.EMBED_MODEL
        except Exception as e:  # gated model without a login, or offline
            print(f"[embed] could not load {config.EMBED_MODEL} ({type(e).__name__}); "
                  f"using fallback {config.EMBED_FALLBACK}", file=sys.stderr)
            self._model = SentenceTransformer(config.EMBED_FALLBACK, device="cpu")
            self.name = config.EMBED_FALLBACK
        prompts = getattr(self._model, "prompts", {}) or {}
        # EmbeddingGemma ships task prompts; "STS" fits description-to-description matching.
        self._prompt_name = "STS" if "STS" in prompts else None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if missing:
            vecs = self._model.encode(missing, prompt_name=self._prompt_name,
                                      normalize_embeddings=True, convert_to_numpy=True)
            for t, v in zip(missing, vecs):
                self._memo[t] = v.astype(np.float32)
        return np.stack([self._memo[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)

    def one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_EMBEDDER: Embedder | None = None


def embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


def best_match(query: str, candidates: list[str]) -> tuple[int, float]:
    """Index and cosine similarity of the candidate closest in meaning to `query`."""
    if not candidates:
        return -1, -1.0
    for i, c in enumerate(candidates):
        if c == query:
            return i, 1.0
    e = embedder()
    sims = e.encode(candidates) @ e.one(query)
    i = int(np.argmax(sims))
    return i, float(sims[i])


def ranked(query: str, candidates: list[str]) -> list[tuple[int, float]]:
    if not candidates:
        return []
    e = embedder()
    sims = e.encode(candidates) @ e.one(query)
    order = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in order]
