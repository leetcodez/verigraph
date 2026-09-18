"""Core domain data models for VeriGraph."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class VerificationStatus(str, Enum):
    """NLI verification status for atomic claims."""
    ENTAILED = "ENTAILED"         # Directly supported by retrieved source context
    NEUTRAL = "NEUTRAL"           # Extrapolated, speculative, or unsupported
    CONTRADICTION = "CONTRADICTION" # Contradicts evidence in retrieved context


class Document(BaseModel):
    """Source document entity."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    filename: str
    mime_type: str = "text/plain"
    file_size_bytes: int = 0
    checksum_sha256: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chunk(BaseModel):
    """Semantic chunk extracted from a document with heading context."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    document_title: str
    chunk_index: int
    content: str
    token_count: int
    heading_breadcrumbs: List[str] = Field(default_factory=list)
    page_number: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Entity(BaseModel):
    """Named entity node for the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    canonical_id: str
    name: str
    entity_type: str = "CONCEPT"  # e.g., ALGORITHM, PROTOCOL, SYSTEM, METRIC, VULNERABILITY
    description: Optional[str] = None
    chunk_ids: List[str] = Field(default_factory=list)


class Triple(BaseModel):
    """Subject-Predicate-Object relation for the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    relation: str
    weight: float = 1.0
    chunk_id: Optional[str] = None
    document_title: Optional[str] = None


class Subgraph(BaseModel):
    """Projected knowledge subgraph for reasoning provenance."""
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class RetrievalCandidate(BaseModel):
    """Scored context chunk candidate from hybrid retrieval."""
    chunk_id: str
    document_title: str
    content: str
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    graph_rank: Optional[int] = None
    rrf_score: float = 0.0
    heading_breadcrumbs: List[str] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    """Individual assertion evaluated by the NLI attribution engine."""
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str
    status: VerificationStatus
    confidence: float
    reasoning: str
    cited_chunk_ids: List[str] = Field(default_factory=list)
    cited_entity_names: List[str] = Field(default_factory=list)
    citation_badge: str = ""


class QueryResult(BaseModel):
    """End-to-end result of a verified query execution."""
    query: str
    response: str
    strategy: str  # e.g. "hybrid_graph_rag", "dense_only", "graph_only"
    retrieved_chunks: List[RetrievalCandidate] = Field(default_factory=list)
    claims: List[ClaimVerification] = Field(default_factory=list)
    faithfulness_score: float = 0.0
    answer_relevance_score: float = 0.0
    latency_ms: int = 0
    subgraph: Subgraph = Field(default_factory=Subgraph)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
