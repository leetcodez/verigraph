"""Integration tests for FastAPI application and pipeline."""

import pytest
from fastapi.testclient import TestClient
from verigraph.api.app import create_app


@pytest.fixture
def client(temp_env):
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "graph_nodes" in data


def test_document_ingestion_and_query_flow(client):
    # 1. Ingest raw text document
    doc_payload = {
        "title": "Raft Consensus Manual",
        "content": (
            "# Raft Consensus\n\n"
            "## Leader Election\n"
            "Raft elects a Leader using randomized election timeouts.\n\n"
            "## Split-Brain Mitigation\n"
            "Raft strictly prevents Split-Brain by requiring a Majority Quorum for all state transitions."
        ),
    }
    ingest_resp = client.post("/api/v1/documents/text", json=doc_payload)
    assert ingest_resp.status_code == 200
    doc_data = ingest_resp.json()
    assert doc_data["title"] == "Raft Consensus Manual"
    assert doc_data["chunk_count"] >= 2

    # 2. List documents
    list_resp = client.get("/api/v1/documents")
    assert list_resp.status_code == 200
    assert list_resp.json()["total_count"] >= 1

    # 3. Synchronous Query
    query_payload = {
        "query": "How does Raft mitigate Split-Brain?",
        "top_k": 3,
        "enable_graph": True,
    }
    query_resp = client.post("/api/v1/query", json=query_payload)
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert "response" in q_data
    assert len(q_data["retrieved_chunks"]) > 0
    assert q_data["faithfulness_score"] > 0.0

    # 4. Knowledge Graph Stats
    graph_resp = client.get("/api/v1/graph/stats")
    assert graph_resp.status_code == 200
    assert graph_resp.json()["total_nodes"] > 0


def test_query_streaming_endpoint(client):
    # Ensure at least one document exists
    client.post(
        "/api/v1/documents/text",
        json={"title": "mTLS Manual", "content": "Mutual TLS enforces Zero Trust encryption between services."},
    )

    query_payload = {"query": "What does Mutual TLS enforce?", "top_k": 2}
    resp = client.post("/api/v1/query/stream", json=query_payload)
    assert resp.status_code == 200
    # Assert response contains SSE events
    body = resp.text
    assert "event: routing" in body
    assert "event: subgraph" in body
    assert "event: token" in body
    assert "event: done" in body
