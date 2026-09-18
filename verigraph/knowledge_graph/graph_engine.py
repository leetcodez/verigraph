"""High-performance NetworkX property graph engine for Knowledge Graph traversal."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
from verigraph.core.models import Entity, Subgraph, Triple
from verigraph.core.logging import logger


class KnowledgeGraphEngine:
    """
    Property Graph Engine powering GraphRAG multi-hop reasoning.
    Maintains entity nodes, typed relationships, and context provenance.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path
        self.graph = nx.MultiDiGraph()
        self._entities: Dict[str, Entity] = {}  # canonical_id -> Entity
        self._triples: Dict[str, Triple] = {}    # triple_id -> Triple

        if persistence_path and persistence_path.exists():
            self.load()

    def add_entity(self, entity: Entity) -> None:
        """Adds or updates an entity node in the graph."""
        c_id = entity.canonical_id.lower()
        if c_id in self._entities:
            existing = self._entities[c_id]
            # Merge chunk references
            merged_chunks = list(set(existing.chunk_ids + entity.chunk_ids))
            existing.chunk_ids = merged_chunks
            self.graph.nodes[c_id]["chunk_ids"] = merged_chunks
        else:
            self._entities[c_id] = entity
            self.graph.add_node(
                c_id,
                id=entity.id,
                name=entity.name,
                type=entity.entity_type,
                description=entity.description or "",
                chunk_ids=entity.chunk_ids,
            )

    def add_triple(self, triple: Triple) -> None:
        """Adds a directed relationship between two entities."""
        src = triple.source_id.lower()
        tgt = triple.target_id.lower()

        # Ensure both nodes exist
        if src not in self._entities:
            self.add_entity(Entity(canonical_id=src, name=triple.source_name))
        if tgt not in self._entities:
            self.add_entity(Entity(canonical_id=tgt, name=triple.target_name))

        self._triples[triple.id] = triple
        self.graph.add_edge(
            src,
            tgt,
            key=triple.id,
            id=triple.id,
            relation=triple.relation,
            weight=triple.weight,
            chunk_id=triple.chunk_id or "",
            document_title=triple.document_title or "",
        )

    def find_matching_entities(self, query_tokens: List[str]) -> List[str]:
        """Finds canonical IDs of entities mentioned or matched in query tokens."""
        matched: Set[str] = set()
        query_text = " ".join(query_tokens).lower()

        for c_id, ent in self._entities.items():
            name_lower = ent.name.lower()
            if name_lower in query_text or c_id in query_tokens:
                matched.add(c_id)
            else:
                # Substring token match
                ent_words = name_lower.split()
                if any(w in query_tokens for w in ent_words if len(w) > 3):
                    matched.add(c_id)

        return list(matched)

    def get_subgraph_for_entities(
        self,
        entity_ids: List[str],
        max_depth: int = 2,
        max_nodes: int = 30,
    ) -> Subgraph:
        """
        Extracts a localized k-hop neighborhood subgraph around seed entities.
        Returns a Subgraph model formatted for frontend visualization and context injection.
        """
        seed_nodes = [e.lower() for e in entity_ids if e.lower() in self.graph]
        if not seed_nodes:
            return Subgraph()

        visited_nodes: Set[str] = set(seed_nodes)
        frontier: Set[str] = set(seed_nodes)

        for _ in range(max_depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                # Successors and Predecessors (both directions)
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for n in neighbors:
                    if n not in visited_nodes and len(visited_nodes) < max_nodes:
                        visited_nodes.add(n)
                        next_frontier.add(n)
            frontier = next_frontier
            if len(visited_nodes) >= max_nodes:
                break

        sub_g = self.graph.subgraph(visited_nodes)

        nodes_data: List[Dict[str, Any]] = []
        for n, d in sub_g.nodes(data=True):
            ent = self._entities.get(n)
            nodes_data.append({
                "id": n,
                "name": d.get("name", n),
                "type": d.get("type", "CONCEPT"),
                "description": d.get("description", ""),
                "chunk_ids": d.get("chunk_ids", []),
                "degree": sub_g.degree(n),
                "is_seed": n in seed_nodes,
            })

        edges_data: List[Dict[str, Any]] = []
        for u, v, key, d in sub_g.edges(keys=True, data=True):
            edges_data.append({
                "id": d.get("id", f"{u}-{v}-{key}"),
                "source": u,
                "target": v,
                "relation": d.get("relation", "RELATED_TO"),
                "weight": d.get("weight", 1.0),
                "chunk_id": d.get("chunk_id", ""),
                "document_title": d.get("document_title", ""),
            })

        return Subgraph(nodes=nodes_data, edges=edges_data)

    def find_relational_path(self, entity_a: str, entity_b: str) -> Optional[List[Dict[str, Any]]]:
        """Finds the shortest relational path between two entities for multi-hop synthesis."""
        src = entity_a.lower()
        tgt = entity_b.lower()

        if src not in self.graph or tgt not in self.graph:
            return None

        try:
            path = nx.shortest_path(self.graph.to_undirected(), source=src, target=tgt)
            # Format path into sequential hops
            hops = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Check edges between u and v
                edge_data = None
                if self.graph.has_edge(u, v):
                    edge_data = list(self.graph.get_edge_data(u, v).values())[0]
                elif self.graph.has_edge(v, u):
                    edge_data = list(self.graph.get_edge_data(v, u).values())[0]

                relation = edge_data.get("relation", "CONNECTS") if edge_data else "CONNECTS"
                hops.append({
                    "from": self._entities[u].name,
                    "to": self._entities[v].name,
                    "relation": relation,
                })
            return hops
        except nx.NetworkXNoPath:
            return None

    def get_central_entities(self, top_k: int = 8) -> List[Dict[str, Any]]:
        """Calculates PageRank or Degree Centrality to highlight key architectural concepts."""
        if len(self.graph) == 0:
            return []
        try:
            scores = nx.pagerank(self.graph, weight="weight")
        except Exception:
            try:
                scores = nx.degree_centrality(self.graph)
            except Exception:
                scores = {n: self.graph.degree(n) for n in self.graph.nodes}

        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "canonical_id": n,
                "name": self._entities[n].name,
                "type": self._entities[n].entity_type,
                "score": round(float(score), 4),
            }
            for n, score in sorted_nodes if n in self._entities
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Returns node and edge statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4) if len(self.graph) > 1 else 0.0,
            "connected_components": nx.number_weakly_connected_components(self.graph) if len(self.graph) > 0 else 0,
        }

    def delete_document(self, document_id: str) -> None:
        """Removes triples and isolated entities originating from a deleted document."""
        triples_to_remove = [t_id for t_id, t in self._triples.items() if t.chunk_id and t.chunk_id.startswith(document_id)]
        for t_id in triples_to_remove:
            t = self._triples.pop(t_id)
            if self.graph.has_edge(t.source_id.lower(), t.target_id.lower(), key=t_id):
                self.graph.remove_edge(t.source_id.lower(), t.target_id.lower(), key=t_id)

        # Remove isolated entities
        isolated = list(nx.isolates(self.graph))
        for n in isolated:
            self.graph.remove_node(n)
            self._entities.pop(n, None)

        self.save()

    def save(self) -> None:
        """Persists the graph to disk."""
        if not self.persistence_path:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": [e.model_dump(mode="json") for e in self._entities.values()],
            "triples": [t.model_dump(mode="json") for t in self._triples.values()],
        }
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Loads graph from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ent_data in data.get("entities", []):
                ent = Entity(**ent_data)
                self.add_entity(ent)
            for tr_data in data.get("triples", []):
                tr = Triple(**tr_data)
                self.add_triple(tr)
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph from disk: {e}")
