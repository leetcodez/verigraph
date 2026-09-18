"""Attribution mapping and inline citation builder."""

from typing import Dict, List, Tuple
from verigraph.core.models import ClaimVerification, RetrievalCandidate


class AttributionBuilder:
    """
    Constructs inline citations, links claims to physical chunks,
    and formats evidence cards for frontend inspection.
    """

    def format_attributed_response(
        self,
        claims: List[ClaimVerification],
        retrieved_contexts: List[RetrievalCandidate],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Builds an attributed markdown output with superscript citation references.
        Returns:
            (formatted_markdown_text, sources_directory)
        """
        chunk_map = {c.chunk_id: c for c in retrieved_contexts}
        citation_indices: Dict[str, int] = {}
        sources: List[Dict[str, str]] = []
        next_citation_idx = 1

        annotated_paragraphs: List[str] = []

        for claim in claims:
            citation_marker = ""
            if claim.cited_chunk_ids:
                ref_nums = []
                for cid in claim.cited_chunk_ids:
                    if cid not in citation_indices:
                        citation_indices[cid] = next_citation_idx
                        chunk = chunk_map.get(cid)
                        sources.append({
                            "index": str(next_citation_idx),
                            "chunk_id": cid,
                            "document_title": chunk.document_title if chunk else "Unknown Document",
                            "breadcrumbs": " > ".join(chunk.heading_breadcrumbs) if chunk else "",
                            "excerpt": chunk.content[:200] + "..." if chunk else "",
                        })
                        next_citation_idx += 1
                    ref_nums.append(str(citation_indices[cid]))

                citation_marker = f" [{', '.join(ref_nums)}]"

            # Append badge depending on verification status
            badge = ""
            if claim.status.value == "ENTAILED":
                badge = " ✓"
            elif claim.status.value == "CONTRADICTION":
                badge = " ⚠️ [Contradiction Detected]"

            annotated_paragraphs.append(f"{claim.claim_text}{citation_marker}{badge}")

        formatted_text = " ".join(annotated_paragraphs)
        return formatted_text, sources
