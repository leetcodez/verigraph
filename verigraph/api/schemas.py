"""Pydantic request and response schemas for REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from verigraph.core.models import (
    ClaimVerification,
    Document,
    QueryResult,
    RetrievalCandidate,
    Subgraph,
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's natural language question or prompt")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context passages to return")
    enable_graph: bool = Field(default=True, description="Whether to include knowledge graph traversal")


class QueryResponse(BaseModel):
    query: str
    response: str
    strategy: str
    faithfulness_score: float
    answer_relevance_score: float
    latency_ms: int
    retrieved_chunks: List[RetrievalCandidate]
    claims: List[ClaimVerification]
    subgraph: Subgraph


class DocumentIngestTextRequest(BaseModel):
    title: str
    content: str


class DocumentListResponse(BaseModel):
    documents: List[Document]
    total_count: int


class GraphStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    density: float
    connected_components: int
    central_entities: List[Dict[str, Any]]


class PathSearchRequest(BaseModel):
    source_entity: str
    target_entity: str


class BenchmarkRunRequest(BaseModel):
    num_samples: int = Field(default=5, ge=1, le=20)
