"""Query intent routing and subquery decomposition."""

import re
from typing import List, Tuple


class QueryRouter:
    """
    Classifies incoming user queries into retrieval archetypes:
    - COMPARISON: Queries comparing two distinct systems/concepts (e.g. 'Raft vs Paxos').
    - MULTI_HOP: Queries asking for indirect causality, prevention, or requirement chains.
    - FACTUAL: Simple definition or single-entity lookup.
    - ARCHITECTURE: High-level system structure or overview.
    """

    COMPARISON_REGEX = re.compile(r"\b(vs|versus|compared\s+to|difference\s+between|compare)\b", re.IGNORECASE)
    MULTI_HOP_REGEX = re.compile(r"\b(how\s+does|why\s+does|prevent|avoid|lead\s+to|cause|rely\s+on)\b", re.IGNORECASE)
    ARCH_REGEX = re.compile(r"\b(architecture|overview|design|components|structure|layers)\b", re.IGNORECASE)

    def route(self, query: str) -> Tuple[str, List[str]]:
        """
        Returns:
            strategy: Archetype name ('comparison', 'multi_hop', 'architecture', 'factual')
            subqueries: List of decomposed sub-queries for broad multi-step retrieval.
        """
        q = query.strip()

        # 1. Comparison check
        if self.COMPARISON_REGEX.search(q):
            # Try to decompose into separate entity queries
            parts = self.COMPARISON_REGEX.split(q)
            subqueries = [p.strip() for p in parts if len(p.strip()) > 3]
            if not subqueries:
                subqueries = [q]
            return "comparison", subqueries

        # 2. Multi-hop causality check
        if self.MULTI_HOP_REGEX.search(q):
            return "multi_hop", [q]

        # 3. Architecture / overview check
        if self.ARCH_REGEX.search(q):
            return "architecture", [q]

        # 4. Standard factual query
        return "factual", [q]
