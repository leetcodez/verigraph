"""Reciprocal Rank Fusion (RRF) for multi-channel hybrid retrieval."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class ReciprocalRankFusion:
    """
    Combines ranked lists from heterogeneous retrieval modalities:
    - Dense Vector Similarity
    - Sparse Lexical BM25
    - Knowledge Graph Subgraph Expansion
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        rankings: Dict[str, List[Tuple[str, float]]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float, Dict[str, int]]]:
        """
        Calculates RRF score across channels.
        
        Args:
            rankings: Map from modality name (e.g. 'dense', 'bm25', 'graph')
                      to ordered list of (item_id, original_score).
            weights: Optional channel weights (default 1.0 each).
            
        Returns:
            List of (item_id, rrf_score, {modality: rank_index}) sorted descending by RRF score.
        """
        weights = weights or {}
        item_scores: Dict[str, float] = defaultdict(float)
        item_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)

        for modality, ranked_items in rankings.items():
            modality_weight = weights.get(modality, 1.0)
            for rank_idx, (item_id, _) in enumerate(ranked_items, start=1):
                item_ranks[item_id][modality] = rank_idx
                # RRF Formula
                item_scores[item_id] += modality_weight / (self.k + rank_idx)

        # Sort descending
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [(item_id, score, item_ranks[item_id]) for item_id, score in sorted_items]
