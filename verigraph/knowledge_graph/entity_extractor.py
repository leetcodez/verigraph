"""Entity and relational triple extraction engine."""

import re
from typing import List, Optional, Set, Tuple
from verigraph.core.models import Chunk, Entity, Triple


class EntityExtractor:
    """
    Extracts canonical entities and Subject-Predicate-Object triples from chunks.
    Combines grammatical heuristics, syntactic dependency patterns, and domain dictionaries.
    """

    # Recognized relationship predicates
    RELATION_PATTERNS = [
        (r"\b(prevents?|mitigates?|eliminates?|avoids?)\b", "PREVENTS"),
        (r"\b(implements?|realizes?|provides?)\b", "IMPLEMENTS"),
        (r"\b(requires?|needs?|depends\s+on)\b", "REQUIRES"),
        (r"\b(elects?|selects?|chooses?)\b", "ELECTS"),
        (r"\b(replicates?|synchronizes?|broadcasts?)\b", "REPLICATES_TO"),
        (r"\b(enforces?|validates?|verifies?)\b", "ENFORCES"),
        (r"\b(contains?|consists\s+of|includes?)\b", "CONTAINS"),
        (r"\b(causes?|leads\s+to|results\s+in)\b", "LEADS_TO"),
        (r"\b(is\s+a|is\s+an|acts\s+as)\b", "IS_A"),
        (r"\b(connects\s+to|communicates\s+with)\b", "COMMUNICATES_WITH"),
    ]

    # Stop entities that shouldn't be added as standalone concepts
    STOP_ENTITIES = {
        "it", "this", "that", "these", "those", "they", "we", "system", "systems",
        "approach", "method", "algorithm", "problem", "solution", "section", "figure",
        "table", "data", "process", "case", "example", "result", "paper", "author"
    }

    def __init__(self):
        self._compiled_relations = [(re.compile(p, re.IGNORECASE), rel) for p, rel in self.RELATION_PATTERNS]
        self._capitalized_phrase = re.compile(r"\b[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\b")
        self._backticked_phrase = re.compile(r"`([^`]+)`")

    def _canonicalize(self, name: str) -> str:
        """Converts an entity name to a canonical slug."""
        normalized = name.replace("-", " ").replace("_", " ")
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", normalized).strip().lower()
        return re.sub(r"\s+", "_", clean)

    def _extract_candidate_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extracts candidate entities: returns (original_name, canonical_id)."""
        candidates: Set[str] = set()

        # 1. Backticked technical terms: `Raft`, `mTLS`
        for term in self._backticked_phrase.findall(text):
            if len(term) >= 2 and term.lower() not in self.STOP_ENTITIES:
                candidates.add(term.strip())

        # 2. Capitalized phrases & Title Case entities
        for match in self._capitalized_phrase.findall(text):
            m_clean = match.strip()
            if len(m_clean) >= 3 and m_clean.lower() not in self.STOP_ENTITIES:
                candidates.add(m_clean)

        # 3. Known technical keywords even if lowercase
        domain_keywords = [
            "raft", "paxos", "multi-paxos", "viewstamped replication", "vector clock",
            "split-brain", "leader election", "log replication", "majority quorum",
            "heartbeat", "term", "candidate", "follower", "leader", "commit index",
            "byzantine fault tolerance", "two-phase commit", "etcd", "zookeeper",
            "oauth2", "jwt", "mtls", "zero trust", "rbac", "api gateway",
            "bm25", "graphrag", "hybrid search", "nli verification", "reranker",
            "qdrant", "neo4j", "fastapi", "networkx", "postgresql", "redis"
        ]
        text_lower = text.lower()
        for kw in domain_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                # Format nicely
                candidates.add(" ".join(w.capitalize() for w in kw.split()))

        # Normalize and filter
        results = []
        for c in candidates:
            c_id = self._canonicalize(c)
            if c_id and c_id not in self.STOP_ENTITIES and len(c_id) >= 2:
                results.append((c, c_id))

        return results

    def extract_from_chunk(self, chunk: Chunk) -> Tuple[List[Entity], List[Triple]]:
        """
        Extracts entities and relational triples from a single chunk.
        """
        entities: List[Entity] = []
        triples: List[Triple] = []

        candidate_ents = self._extract_candidate_entities(chunk.content)
        ent_map = {c_id: name for name, c_id in candidate_ents}

        for name, c_id in candidate_ents:
            # Determine coarse entity type
            etype = "CONCEPT"
            c_lower = c_id.lower()
            if any(k in c_lower for k in ["raft", "paxos", "bm25", "bft"]):
                etype = "ALGORITHM"
            elif any(k in c_lower for k in ["oauth", "jwt", "mtls", "http", "grpc"]):
                etype = "PROTOCOL"
            elif any(k in c_lower for k in ["neo4j", "qdrant", "redis", "postgres", "fastapi"]):
                etype = "SYSTEM"
            elif any(k in c_lower for k in ["split_brain", "partition", "vulnerability", "attack"]):
                etype = "VULNERABILITY"
            elif any(k in c_lower for k in ["leader", "follower", "candidate"]):
                etype = "ROLE"

            entities.append(
                Entity(
                    canonical_id=c_id,
                    name=name,
                    entity_type=etype,
                    chunk_ids=[chunk.id],
                )
            )

        # Sentence-level relation extraction
        sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
        for sent in sentences:
            sent_lower = sent.lower()

            # Find which extracted entities are in this sentence
            present_ents = [
                (name, c_id) for c_id, name in ent_map.items()
                if name.lower() in sent_lower or c_id.replace("_", " ") in sent_lower
            ]

            if len(present_ents) < 2:
                continue

            # Check if any predicate matches in this sentence
            for regex, rel in self._compiled_relations:
                match = regex.search(sent_lower)
                if match:
                    pred_start = match.start()

                    # Find entity before predicate and entity after predicate
                    before_ents = [
                        (name, c_id) for name, c_id in present_ents
                        if sent_lower.find(name.lower()) < pred_start
                    ]
                    after_ents = [
                        (name, c_id) for name, c_id in present_ents
                        if sent_lower.find(name.lower()) > pred_start
                    ]

                    if before_ents and after_ents:
                        src_name, src_id = before_ents[-1]
                        tgt_name, tgt_id = after_ents[0]

                        if src_id != tgt_id:
                            triple = Triple(
                                source_id=src_id,
                                source_name=src_name,
                                target_id=tgt_id,
                                target_name=tgt_name,
                                relation=rel,
                                weight=1.0,
                                chunk_id=chunk.id,
                                document_title=chunk.document_title,
                            )
                            triples.append(triple)

        return entities, triples
