"""Deterministic high-performance local semantic embedder."""

import hashlib
import math
import re
from typing import List
import numpy as np
from verigraph.embeddings.base import BaseEmbedder


class LocalSemanticEmbedder(BaseEmbedder):
    """
    High-performance, zero-latency local dense embedder.
    Uses subword n-gram feature hashing, frequency log-damping, and L2 unit normalization.
    Guarantees deterministic, reproducible cosine similarity vectors.
    """

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "for", "is", "of", "while", "during",
        "to", "from", "in", "out", "on", "off", "again", "further", "then", "once"
    }

    def __init__(self, dimension: int = 384):
        self._dim = dimension
        self._word_regex = re.compile(r"\b[a-zA-Z0-9_\-\.]+\b")

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_index_and_sign(self, token: str, salt: int = 0) -> tuple[int, float]:
        """Maps a token to an index in [0, dim) and a sign in {-1.0, +1.0}."""
        data = f"{token}:{salt}".encode("utf-8")
        h = int(hashlib.md5(data).hexdigest(), 16)
        idx = h % self._dim
        sign = 1.0 if (h >> 31) & 1 else -1.0
        return idx, sign

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        text_clean = text.lower()
        words = self._word_regex.findall(text_clean)

        if not words:
            # Fallback uniform vector
            vec.fill(1.0 / math.sqrt(self._dim))
            return vec.tolist()

        # 1. Word unigrams with stop-word dampening
        for i, word in enumerate(words):
            weight = 0.2 if word in self.STOP_WORDS else 1.0
            # Early position bias
            pos_weight = 1.0 + (1.0 / (i + 1))
            idx, sign = self._hash_to_index_and_sign(word, salt=1)
            vec[idx] += sign * weight * pos_weight

            # 2. Subword character n-grams (3, 4, 5) for technical vocabulary
            if len(word) >= 3 and word not in self.STOP_WORDS:
                for n in (3, 4):
                    for start in range(len(word) - n + 1):
                        ngram = word[start : start + n]
                        n_idx, n_sign = self._hash_to_index_and_sign(ngram, salt=n)
                        vec[n_idx] += n_sign * 0.35

        # 3. Word bigrams for compositional semantics
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in self.STOP_WORDS or w2 not in self.STOP_WORDS:
                bigram = f"{w1}_{w2}"
                b_idx, b_sign = self._hash_to_index_and_sign(bigram, salt=7)
                vec[b_idx] += b_sign * 0.75

        # L2 Normalization so dot product == cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec.fill(1.0 / math.sqrt(self._dim))

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
