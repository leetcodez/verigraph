"""Cross-relevance reranker with structural and entity weighting."""

import re
from typing import List, Set
from verigraph.core.models import RetrievalCandidate


class RelevanceReranker:
    """
    Reranks candidates produced by RRF fusion using contextual signals:
    - Heading breadcrumb relevance
    - Entity density
    - Query term co-occurrence
    - Content completeness
    """

    def __init__(self):
        self._word_regex = re.compile(r"\b[a-zA-Z0-9_\-\.]+\b")

    def _tokenize(self, text: str) -> Set[str]:
        return {w.lower() for w in self._word_regex.findall(text) if len(w) > 2}

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Reranks and trims candidate chunks.
        """
        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return candidates[:top_k]

        scored: List[tuple[RetrievalCandidate, float]] = []

        for cand in candidates:
            chunk_tokens = self._tokenize(cand.content)
            breadcrumbs_text = " ".join(cand.heading_breadcrumbs).lower()

            # 1. Breadcrumb match boost (title relevance)
            breadcrumb_overlap = sum(1 for q in query_tokens if q in breadcrumbs_text)
            breadcrumb_bonus = 0.15 * min(breadcrumb_overlap, 3)

            # 2. Query token recall in content
            overlap_count = len(query_tokens & chunk_tokens)
            recall_score = overlap_count / max(len(query_tokens), 1)

            # 3. Base RRF contribution
            base_score = cand.rrf_score * 10.0

            # 4. Modality cross-verification bonus (found in all 3 modalities)
            modalities_count = sum(
                1 for r in [cand.dense_rank, cand.bm25_rank, cand.graph_rank] if r is not None
            )
            multi_channel_bonus = 0.25 * (modalities_count - 1) if modalities_count > 1 else 0.0

            final_score = base_score + (recall_score * 0.4) + breadcrumb_bonus + multi_channel_bonus
            cand.score = round(final_score, 4)
            scored.append((cand, final_score))

        # Sort descending by final score
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
