"""Embeddings package factory."""

from verigraph.config import settings
from verigraph.embeddings.base import BaseEmbedder
from verigraph.embeddings.local_embedder import LocalSemanticEmbedder
from verigraph.embeddings.remote_embedder import RemoteEmbedder


def get_embedder() -> BaseEmbedder:
    """Returns the configured embedder instance."""
    if settings.embedding_model == "openai" and (settings.openai_api_key or settings.openai_base_url):
        return RemoteEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            dimension=settings.embedding_dim,
        )
    return LocalSemanticEmbedder(dimension=settings.embedding_dim)


__all__ = ["BaseEmbedder", "LocalSemanticEmbedder", "RemoteEmbedder", "get_embedder"]
