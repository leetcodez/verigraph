"""BM25Okapi lexical inverted index for sparse keyword retrieval."""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
from verigraph.core.models import Chunk


class BM25Index:
    """
    In-memory Okapi BM25 index.
    Provides term-exact and keyword retrieval to complement dense embeddings in hybrid search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count: int = 0
        self.avgdl: float = 0.0
        
        # doc_id -> list of tokens
        self.doc_lengths: Dict[str, int] = {}
        # doc_id -> Chunk
        self.chunks: Dict[str, Chunk] = {}
        # term -> set of chunk_ids
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        # (term, chunk_id) -> term frequency
        self.term_frequencies: Dict[Tuple[str, str], int] = {}
        
        self._tokenizer_regex = re.compile(r"\b[a-zA-Z0-9_\-\.]+\b")

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text."""
        return [t.lower() for t in self._tokenizer_regex.findall(text)]

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Indexes a batch of chunks into the BM25 inverted index."""
        total_length = self.avgdl * self.doc_count

        for chunk in chunks:
            # If already exists, remove old entry first
            if chunk.id in self.chunks:
                self.remove_chunk(chunk.id)

            # Include breadcrumbs and content in lexical representation
            breadcrumbs_text = " ".join(chunk.heading_breadcrumbs)
            text = f"{chunk.document_title} {breadcrumbs_text} {chunk.content}"
            tokens = self._tokenize(text)
            
            length = len(tokens)
            self.chunks[chunk.id] = chunk
            self.doc_lengths[chunk.id] = length
            total_length += length
            self.doc_count += 1

            # Count term frequencies
            tf_counter = Counter(tokens)
            for term, count in tf_counter.items():
                self.inverted_index[term].add(chunk.id)
                self.term_frequencies[(term, chunk.id)] = count

        if self.doc_count > 0:
            self.avgdl = total_length / self.doc_count

    def remove_chunk(self, chunk_id: str) -> None:
        """Removes a chunk from the BM25 index."""
        if chunk_id not in self.chunks:
            return

        old_len = self.doc_lengths.pop(chunk_id, 0)
        del self.chunks[chunk_id]
        
        # Clean up inverted index & tf table
        terms_to_delete = []
        for (term, c_id) in list(self.term_frequencies.keys()):
            if c_id == chunk_id:
                del self.term_frequencies[(term, chunk_id)]
                if chunk_id in self.inverted_index[term]:
                    self.inverted_index[term].remove(chunk_id)
                    if not self.inverted_index[term]:
                        terms_to_delete.append(term)

        for term in terms_to_delete:
            del self.inverted_index[term]

        self.doc_count -= 1
        if self.doc_count > 0:
            total_len = (self.avgdl * (self.doc_count + 1)) - old_len
            self.avgdl = total_len / self.doc_count
        else:
            self.avgdl = 0.0

    def remove_document(self, document_id: str) -> None:
        """Removes all chunks associated with a document_id."""
        matching_ids = [c.id for c in self.chunks.values() if c.document_id == document_id]
        for cid in matching_ids:
            self.remove_chunk(cid)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Executes BM25Okapi scoring against the inverted index.
        Returns: List of (Chunk, bm25_score) sorted in descending order.
        """
        if self.doc_count == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Find candidate documents containing at least one query term
        candidate_ids: Set[str] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_ids.update(self.inverted_index[token])

        if not candidate_ids:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            matching_docs = self.inverted_index[token]
            df = len(matching_docs)

            # Robertson-Spärck Jones IDF
            idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            idf = max(idf, 0.01)

            for doc_id in matching_docs:
                tf = self.term_frequencies.get((token, doc_id), 0)
                doc_len = self.doc_lengths.get(doc_id, self.avgdl)

                # BM25 formula term
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                scores[doc_id] += idf * (numerator / denominator)

        # Sort and return top_k
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.chunks[doc_id], score) for doc_id, score in sorted_docs]
