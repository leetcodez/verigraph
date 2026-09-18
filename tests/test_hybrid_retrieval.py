"""Tests for Hybrid Retrieval and RRF."""

from verigraph.core.models import RetrievalCandidate
from verigraph.retrieval.rrf import ReciprocalRankFusion
from verigraph.retrieval.reranker import RelevanceReranker


def test_rrf_scoring():
    rrf = ReciprocalRankFusion(k=60)
    rankings = {
        "dense": [("chunk1", 0.9), ("chunk2", 0.8)],
        "bm25": [("chunk2", 12.0), ("chunk3", 8.0)],
        "graph": [("chunk2", 2.0)],
    }
    # chunk2 is ranked in all 3 channels, so its RRF score must be the highest
    results = rrf.fuse(rankings)
    assert len(results) == 3
    top_chunk_id, top_score, ranks = results[0]
    assert top_chunk_id == "chunk2"
    assert "dense" in ranks
    assert "bm25" in ranks
    assert "graph" in ranks


def test_relevance_reranker():
    reranker = RelevanceReranker()
    candidates = [
        RetrievalCandidate(
            chunk_id="c1",
            document_title="Doc A",
            content="General consensus overview in networks.",
            score=0.1,
            rrf_score=0.01,
        ),
        RetrievalCandidate(
            chunk_id="c2",
            document_title="Doc B",
            content="Raft prevents split-brain using majority quorum during partitions.",
            score=0.1,
            rrf_score=0.01,
            heading_breadcrumbs=["Distributed Systems", "Split-Brain"],
        ),
    ]

    reranked = reranker.rerank("How does Raft avoid split-brain?", candidates, top_k=2)
    assert len(reranked) == 2
    # c2 must rank higher due to keyword recall and breadcrumb alignment
    assert reranked[0].chunk_id == "c2"
