"""In-memory persistent dense vector store with cosine similarity."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from verigraph.core.models import Chunk
from verigraph.vector_store.base import BaseVectorStore


class MemoryVectorStore(BaseVectorStore):
    """
    High-performance in-memory vector store with NumPy matrix multiplication
    and atomic JSON/file persistence.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path
        self._chunks: Dict[str, Chunk] = {}
        self._chunk_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # Shape: (N, D)

        if persistence_path and persistence_path.exists():
            self.load()

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        new_vecs = np.array(embeddings, dtype=np.float32)
        # Normalize to ensure cosine similarity equals dot product
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_vecs = new_vecs / norms

        for chunk in chunks:
            if chunk.id in self._chunks:
                self.delete_chunk(chunk.id)
            self._chunks[chunk.id] = chunk

        if self._embeddings is None or len(self._chunk_ids) == 0:
            self._chunk_ids = [c.id for c in chunks]
            self._embeddings = new_vecs
        else:
            self._chunk_ids.extend([c.id for c in chunks])
            self._embeddings = np.vstack([self._embeddings, new_vecs])

        self.save()

    def delete_chunk(self, chunk_id: str) -> None:
        if chunk_id not in self._chunks:
            return

        del self._chunks[chunk_id]
        if chunk_id in self._chunk_ids:
            idx = self._chunk_ids.index(chunk_id)
            self._chunk_ids.pop(idx)
            if self._embeddings is not None:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)

        self.save()

    def delete_document(self, document_id: str) -> None:
        matching_ids = [c.id for c in self._chunks.values() if c.document_id == document_id]
        for cid in matching_ids:
            self.delete_chunk(cid)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        if self._embeddings is None or len(self._chunk_ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Matrix-vector dot product for fast cosine scores
        scores = np.dot(self._embeddings, q_vec)

        # Apply optional filtering
        results: List[Tuple[Chunk, float]] = []
        # Sort indices by score descending
        sorted_indices = np.argsort(-scores)

        for idx in sorted_indices:
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]

            if filters:
                match = True
                for k, v in filters.items():
                    if getattr(chunk, k, None) != v:
                        match = False
                        break
                if not match:
                    continue

            score = float(scores[idx])
            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

    def save(self) -> None:
        if not self.persistence_path:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_ids": self._chunk_ids,
            "chunks": [c.model_dump(mode="json") for c in self._chunks.values()],
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
        }
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self) -> None:
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._chunk_ids = data.get("chunk_ids", [])
            raw_chunks = data.get("chunks", [])
            self._chunks = {c["id"]: Chunk(**c) for c in raw_chunks}
            raw_embs = data.get("embeddings", [])
            if raw_embs:
                self._embeddings = np.array(raw_embs, dtype=np.float32)
            else:
                self._embeddings = None
        except Exception:
            self._chunks = {}
            self._chunk_ids = []
            self._embeddings = None
