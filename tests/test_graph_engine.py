"""Tests for Property Graph Engine and Entity Extraction."""

from verigraph.core.models import Chunk, Entity, Triple
from verigraph.knowledge_graph.graph_engine import KnowledgeGraphEngine
from verigraph.knowledge_graph.entity_extractor import EntityExtractor


def test_entity_extractor():
    extractor = EntityExtractor()
    chunk = Chunk(
        document_id="doc1",
        document_title="Test Doc",
        chunk_index=0,
        content="Raft prevents split-brain during network partitions. The Leader replicates logs to Followers.",
        token_count=15,
    )
    entities, triples = extractor.extract_from_chunk(chunk)

    assert len(entities) > 0
    canonical_ids = [e.canonical_id for e in entities]
    assert "raft" in canonical_ids
    assert "split_brain" in canonical_ids

    # Verify relation extraction
    relations = [t.relation for t in triples]
    assert "PREVENTS" in relations or "REPLICATES_TO" in relations


def test_graph_engine_neighborhood_and_pagerank():
    engine = KnowledgeGraphEngine()

    # Add entities and relations
    e1 = Entity(canonical_id="raft", name="Raft", entity_type="ALGORITHM")
    e2 = Entity(canonical_id="split_brain", name="Split Brain", entity_type="VULNERABILITY")
    e3 = Entity(canonical_id="leader", name="Leader", entity_type="ROLE")

    engine.add_entity(e1)
    engine.add_entity(e2)
    engine.add_entity(e3)

    engine.add_triple(
        Triple(
            source_id="raft",
            source_name="Raft",
            target_id="split_brain",
            target_name="Split Brain",
            relation="PREVENTS",
        )
    )
    engine.add_triple(
        Triple(
            source_id="raft",
            source_name="Raft",
            target_id="leader",
            target_name="Leader",
            relation="ELECTS",
        )
    )

    # Test Subgraph expansion
    subgraph = engine.get_subgraph_for_entities(["raft"], max_depth=1)
    assert len(subgraph.nodes) == 3
    assert len(subgraph.edges) == 2

    # Test Shortest path
    path = engine.find_relational_path("leader", "split_brain")
    assert path is not None
    assert len(path) == 2

    # Test PageRank
    central = engine.get_central_entities(top_k=3)
    assert len(central) > 0
    assert any(c["canonical_id"] == "raft" for c in central)
