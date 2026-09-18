"""Base LLM synthesizer interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional
from verigraph.core.models import RetrievalCandidate, Subgraph


class BaseSynthesizer(ABC):
    """Abstract interface for multi-hop synthesis."""

    @abstractmethod
    def generate(
        self,
        query: str,
        retrieved_contexts: List[RetrievalCandidate],
        subgraph: Optional[Subgraph] = None,
    ) -> str:
        """Generates a synthesized answer based on retrieved context and knowledge graph."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        query: str,
        retrieved_contexts: List[RetrievalCandidate],
        subgraph: Optional[Subgraph] = None,
    ) -> AsyncGenerator[str, None]:
        """Streams generated response tokens asynchronously."""
        pass
