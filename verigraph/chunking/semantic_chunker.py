"""Hierarchical and semantic document chunker."""

import re
from typing import List, Tuple
from verigraph.core.models import Chunk


class SemanticChunker:
    """
    Splits structured sections into semantic chunks while:
    1. Preserving sentence boundaries.
    2. Injecting hierarchical heading breadcrumbs into the context.
    3. Applying sliding token overlaps to avoid boundary knowledge loss.
    """

    def __init__(self, target_chunk_size: int = 384, chunk_overlap: int = 64, min_chunk_size: int = 40):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._sentence_regex = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token count estimation (~4 characters per token or whitespace split)."""
        words = text.split()
        return max(len(words), int(len(text) / 4))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text cleanly into sentences."""
        sentences = self._sentence_regex.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_sections(
        self,
        document_id: str,
        document_title: str,
        sections: List[Tuple[str, List[str], int]],
    ) -> List[Chunk]:
        """
        Processes list of (section_text, heading_breadcrumbs, page_number)
        and returns a list of enriched Chunk objects.
        """
        chunks: List[Chunk] = []
        chunk_idx = 0

        for section_text, breadcrumbs, page_num in sections:
            sentences = self._split_into_sentences(section_text)
            if not sentences:
                continue

            current_sentences: List[str] = []
            current_token_count = 0

            for sent in sentences:
                sent_tokens = self._estimate_tokens(sent)

                # If adding this sentence exceeds target size and we have accumulated enough
                if current_token_count + sent_tokens > self.target_chunk_size and current_token_count >= self.min_chunk_size:
                    chunk_text = " ".join(current_sentences)
                    breadcrumb_prefix = " > ".join(breadcrumbs) if breadcrumbs else ""
                    
                    # Store chunk with enriched breadcrumb context
                    chunk = Chunk(
                        document_id=document_id,
                        document_title=document_title,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        token_count=current_token_count,
                        heading_breadcrumbs=breadcrumbs,
                        page_number=page_num,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                    # Sliding overlap: keep the tail sentences that fit within chunk_overlap
                    overlap_sentences: List[str] = []
                    overlap_tokens = 0
                    for s in reversed(current_sentences):
                        st = self._estimate_tokens(s)
                        if overlap_tokens + st <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += st
                        else:
                            break

                    current_sentences = overlap_sentences + [sent]
                    current_token_count = overlap_tokens + sent_tokens
                else:
                    current_sentences.append(sent)
                    current_token_count += sent_tokens

            # Flush remaining sentences in section
            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                if len(chunk_text.split()) >= 4:
                    chunk = Chunk(
                        document_id=document_id,
                        document_title=document_title,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                        token_count=current_token_count,
                        heading_breadcrumbs=breadcrumbs,
                        page_number=page_num,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

        return chunks
