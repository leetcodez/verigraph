"""Base vector store interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from verigraph.core.models import Chunk


class BaseVectorStore(ABC):
    """Abstract interface for vector similarity databases."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Insert or upsert chunks along with their dense embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform cosine similarity search.
        Returns: List of (Chunk, cosine_similarity_score).
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by its unique ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored chunks."""
        pass
