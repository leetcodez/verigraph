"""Tests for Vector Store and BM25 Okapi Index."""

from verigraph.embeddings.local_embedder import LocalSemanticEmbedder
from verigraph.vector_store.memory_store import MemoryVectorStore
from verigraph.vector_store.bm25_index import BM25Index


def test_local_embedder_properties():
    embedder = LocalSemanticEmbedder(dimension=128)
    vec1 = embedder.embed_text("Raft leader election in distributed systems")
    vec2 = embedder.embed_text("Raft consensus leader node")
    vec3 = embedder.embed_text("Pizza pasta italian cuisine")

    assert len(vec1) == 128
    assert len(vec2) == 128

    import numpy as np
    dot12 = np.dot(vec1, vec2)
    dot13 = np.dot(vec1, vec3)

    # Raft texts should be significantly closer to each other than to Pizza
    assert dot12 > dot13


def test_bm25_exact_and_keyword_search(sample_chunks):
    bm25 = BM25Index()
    bm25.add_chunks(sample_chunks)

    # Search for Raft
    results = bm25.search("quorum split-brain", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "split-brain" in top_chunk.content
    assert score > 0.0

    # Search for mTLS
    sec_results = bm25.search("mTLS cryptographic transport", top_k=2)
    assert len(sec_results) > 0
    assert "Mutual TLS" in sec_results[0][0].content


def test_memory_vector_store(sample_chunks):
    embedder = LocalSemanticEmbedder(dimension=64)
    store = MemoryVectorStore()

    texts = [c.content for c in sample_chunks]
    embeddings = embedder.embed_batch(texts)
    store.add_chunks(sample_chunks, embeddings)

    assert store.count() == 3

    q_vec = embedder.embed_text("cryptographic certificates and TLS")
    results = store.search(q_vec, top_k=1)
    assert len(results) == 1
    assert "Mutual TLS" in results[0][0].content
