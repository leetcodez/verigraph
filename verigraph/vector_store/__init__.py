"""Vector store package factory."""

from verigraph.config import settings
from verigraph.vector_store.base import BaseVectorStore
from verigraph.vector_store.memory_store import MemoryVectorStore
from verigraph.vector_store.bm25_index import BM25Index


def get_vector_store() -> BaseVectorStore:
    """Returns a persistent vector store instance."""
    store_file = settings.storage_dir / "vector_store.json"
    return MemoryVectorStore(persistence_path=store_file)


__all__ = ["BaseVectorStore", "MemoryVectorStore", "BM25Index", "get_vector_store"]
