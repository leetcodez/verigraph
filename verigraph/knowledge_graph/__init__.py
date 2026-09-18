"""Knowledge Graph package factory."""

from verigraph.config import settings
from verigraph.knowledge_graph.graph_engine import KnowledgeGraphEngine
from verigraph.knowledge_graph.entity_extractor import EntityExtractor


def get_graph_engine() -> KnowledgeGraphEngine:
    """Returns the persistent graph engine instance."""
    graph_file = settings.storage_dir / "knowledge_graph.json"
    return KnowledgeGraphEngine(persistence_path=graph_file)


__all__ = ["KnowledgeGraphEngine", "EntityExtractor", "get_graph_engine"]
