"""Hybrid retrieval orchestrator combining Vector, BM25, and Graph modalities."""

from typing import Dict, List, Tuple
from verigraph.config import settings
from verigraph.core.models import Chunk, RetrievalCandidate, Subgraph
from verigraph.core.logging import logger
from verigraph.embeddings.base import BaseEmbedder
from verigraph.vector_store.base import BaseVectorStore
from verigraph.vector_store.bm25_index import BM25Index
from verigraph.knowledge_graph.graph_engine import KnowledgeGraphEngine
from verigraph.retrieval.rrf import ReciprocalRankFusion
from verigraph.retrieval.reranker import RelevanceReranker


class HybridRetriever:
    """
    Coordinates simultaneous sparse, dense, and graph-traversal retrieval
    and merges them via Reciprocal Rank Fusion and contextual reranking.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        bm25_index: BM25Index,
        graph_engine: KnowledgeGraphEngine,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.graph_engine = graph_engine
        self.rrf = ReciprocalRankFusion(k=settings.rrf_k)
        self.reranker = RelevanceReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 8,
        bm25_top_k: int = 8,
        graph_top_k: int = 8,
    ) -> Tuple[List[RetrievalCandidate], Subgraph]:
        """
        Executes parallel hybrid retrieval across Dense, BM25, and Knowledge Graph channels.
        """
        rankings: Dict[str, List[Tuple[str, float]]] = {}
        chunk_map: Dict[str, Chunk] = {}

        # 1. Dense Vector Search
        try:
            query_vec = self.embedder.embed_text(query)
            dense_results = self.vector_store.search(query_vec, top_k=dense_top_k)
            rankings["dense"] = [(chunk.id, score) for chunk, score in dense_results]
            for chunk, _ in dense_results:
                chunk_map[chunk.id] = chunk
        except Exception as e:
            logger.warning(f"Dense retrieval error: {e}")
            rankings["dense"] = []

        # 2. Sparse BM25 Search
        try:
            bm25_results = self.bm25_index.search(query, top_k=bm25_top_k)
            rankings["bm25"] = [(chunk.id, score) for chunk, score in bm25_results]
            for chunk, _ in bm25_results:
                chunk_map[chunk.id] = chunk
        except Exception as e:
            logger.warning(f"BM25 retrieval error: {e}")
            rankings["bm25"] = []

        # 3. Knowledge Graph Subgraph Expansion
        query_words = query.lower().split()
        matched_entities = self.graph_engine.find_matching_entities(query_words)
        subgraph = Subgraph()

        if matched_entities:
            subgraph = self.graph_engine.get_subgraph_for_entities(
                matched_entities,
                max_depth=settings.graph_max_hop_depth,
                max_nodes=25,
            )
            # Collect chunk references linked to graph nodes & edges
            graph_chunk_scores: Dict[str, float] = {}
            for node in subgraph.nodes:
                for c_id in node.get("chunk_ids", []):
                    graph_chunk_scores[c_id] = graph_chunk_scores.get(c_id, 0.0) + (2.0 if node.get("is_seed") else 1.0)

            for edge in subgraph.edges:
                c_id = edge.get("chunk_id")
                if c_id:
                    graph_chunk_scores[c_id] = graph_chunk_scores.get(c_id, 0.0) + 1.5

            sorted_graph_chunks = sorted(graph_chunk_scores.items(), key=lambda x: x[1], reverse=True)[:graph_top_k]
            rankings["graph"] = sorted_graph_chunks

            # Fetch any chunks from vector store that were discovered purely via graph
            for c_id, _ in sorted_graph_chunks:
                if c_id not in chunk_map:
                    found_chunk = self.vector_store.get_chunk(c_id)
                    if found_chunk:
                        chunk_map[c_id] = found_chunk
        else:
            rankings["graph"] = []

        # 4. Multi-Channel Reciprocal Rank Fusion
        weights = {"dense": 1.0, "bm25": 1.0, "graph": 1.25}
        fused_results = self.rrf.fuse(rankings, weights=weights)

        # 5. Materialize candidate objects
        candidates: List[RetrievalCandidate] = []
        for chunk_id, rrf_score, ranks in fused_results:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue

            cand = RetrievalCandidate(
                chunk_id=chunk.id,
                document_title=chunk.document_title,
                content=chunk.content,
                score=rrf_score,
                dense_rank=ranks.get("dense"),
                bm25_rank=ranks.get("bm25"),
                graph_rank=ranks.get("graph"),
                rrf_score=rrf_score,
                heading_breadcrumbs=chunk.heading_breadcrumbs,
            )
            candidates.append(cand)

        # 6. Apply Contextual Reranking
        final_candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        return final_candidates, subgraph
