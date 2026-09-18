"""End-to-End VeriGraph Orchestration Pipeline."""

import hashlib
import json
from pathlib import Path
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from verigraph.config import settings
from verigraph.core.logging import logger
from verigraph.core.models import (
    Chunk,
    Document,
    QueryResult,
    RetrievalCandidate,
    Subgraph,
    VerificationStatus,
)
from verigraph.parsers import get_parser_for_file
from verigraph.chunking.semantic_chunker import SemanticChunker
from verigraph.embeddings import get_embedder
from verigraph.vector_store import get_vector_store, BM25Index
from verigraph.knowledge_graph import get_graph_engine, EntityExtractor
from verigraph.retrieval.hybrid_retriever import HybridRetriever
from verigraph.verification.claim_splitter import ClaimSplitter
from verigraph.verification.nli_verifier import NLIVerifier
from verigraph.verification.attribution import AttributionBuilder
from verigraph.llm.synthesis_engine import SynthesisEngine
from verigraph.engine.query_router import QueryRouter


class VeriGraphPipeline:
    """
    Central coordinator orchestrating:
    Ingestion -> Chunking -> Vector/BM25/Graph Indexing -> Hybrid Retrieval ->
    Synthesis -> NLI Verification -> Provenance & Attribution Mapping.
    """

    def __init__(self):
        self.settings = settings
        self.settings.init_directories()

        # Document Registry
        self.docs_registry_path = self.settings.storage_dir / "documents_registry.json"
        self._documents: Dict[str, Document] = {}
        self._load_documents_registry()

        # Subsystems
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.bm25_index = BM25Index()
        self.graph_engine = get_graph_engine()
        self.chunker = SemanticChunker(
            target_chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            min_chunk_size=self.settings.min_chunk_size,
        )
        self.entity_extractor = EntityExtractor()
        self.router = QueryRouter()
        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            graph_engine=self.graph_engine,
        )
        self.synthesizer = SynthesisEngine()
        self.claim_splitter = ClaimSplitter(min_words=self.settings.min_claim_words)
        self.nli_verifier = NLIVerifier(embedder=self.embedder)
        self.attribution_builder = AttributionBuilder()

        # Re-index any persistent chunks into BM25 on startup
        self._reindex_bm25_from_vector_store()

    def _load_documents_registry(self) -> None:
        if self.docs_registry_path.exists():
            try:
                with open(self.docs_registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._documents = {d["id"]: Document(**d) for d in data}
            except Exception as e:
                logger.warning(f"Failed to load document registry: {e}")
                self._documents = {}

    def _save_documents_registry(self) -> None:
        self.docs_registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = [d.model_dump(mode="json") for d in self._documents.values()]
        with open(self.docs_registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _reindex_bm25_from_vector_store(self) -> None:
        """Hydrates BM25 index with stored chunks on startup."""
        if hasattr(self.vector_store, "_chunks") and self.vector_store._chunks:
            all_chunks = list(self.vector_store._chunks.values())
            self.bm25_index.add_chunks(all_chunks)
            logger.info(f"Hydrated BM25 index with {len(all_chunks)} chunks.")

    def ingest_document(self, file_path: Path, title: Optional[str] = None) -> Document:
        """
        Ingests a document through parsing, chunking, embedding, and knowledge graph extraction.
        """
        logger.info(f"Starting ingestion for: {file_path.name}")
        file_bytes = file_path.read_bytes()
        checksum = hashlib.sha256(file_bytes).hexdigest()

        # Check if already ingested
        for doc in self._documents.values():
            if doc.checksum_sha256 == checksum:
                logger.info(f"Document already ingested: {doc.title} ({doc.id})")
                return doc

        doc_title = title or file_path.stem.replace("_", " ").title()
        doc = Document(
            title=doc_title,
            filename=file_path.name,
            mime_type="application/pdf" if file_path.suffix.lower() == ".pdf" else "text/markdown",
            file_size_bytes=len(file_bytes),
            checksum_sha256=checksum,
        )

        # 1. Parse into structured sections
        parser = get_parser_for_file(file_path)
        _, sections = parser.parse(file_path)

        # 2. Semantic and hierarchical chunking
        chunks = self.chunker.chunk_sections(
            document_id=doc.id,
            document_title=doc.title,
            sections=sections,
        )
        doc.chunk_count = len(chunks)

        if chunks:
            # 3. Dense embeddings
            texts = [f"{c.document_title}: {c.content}" for c in chunks]
            embeddings = self.embedder.embed_batch(texts)
            self.vector_store.add_chunks(chunks, embeddings)

            # 4. Sparse BM25 Indexing
            self.bm25_index.add_chunks(chunks)

            # 5. Knowledge Graph Entity & Relation Extraction
            for chunk in chunks:
                entities, triples = self.entity_extractor.extract_from_chunk(chunk)
                for ent in entities:
                    self.graph_engine.add_entity(ent)
                for trip in triples:
                    self.graph_engine.add_triple(trip)

            self.graph_engine.save()

        self._documents[doc.id] = doc
        self._save_documents_registry()
        logger.info(f"Successfully ingested '{doc.title}': {len(chunks)} chunks, graph updated.")
        return doc

    def list_documents(self) -> List[Document]:
        return list(self._documents.values())

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self._documents:
            return False

        doc = self._documents.pop(document_id)
        self.vector_store.delete_document(document_id)
        self.bm25_index.remove_document(document_id)
        self.graph_engine.delete_document(document_id)
        self._save_documents_registry()
        logger.info(f"Deleted document: {doc.title}")
        return True

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        enable_graph: bool = True,
    ) -> QueryResult:
        """
        Executes an end-to-end verified query against the knowledge base.
        """
        start_time = time.perf_counter()
        strategy, subqueries = self.router.route(query_text)

        # 1. Hybrid Retrieval
        candidates, subgraph = self.hybrid_retriever.retrieve(
            query=query_text,
            top_k=top_k,
        )

        if not enable_graph:
            subgraph = Subgraph()

        # 2. Multi-hop Synthesis
        raw_response = self.synthesizer.generate(
            query=query_text,
            retrieved_contexts=candidates,
            subgraph=subgraph,
        )

        # 3. NLI Claim Verification
        claims = self.claim_splitter.split_into_claims(raw_response)
        verifications, faithfulness = self.nli_verifier.verify_response(claims, candidates)

        # 4. Attribution and citation linking
        annotated_response, _ = self.attribution_builder.format_attributed_response(
            verifications, candidates
        )

        # Answer relevance metric (query-response cosine similarity)
        q_vec = self.embedder.embed_text(query_text)
        r_vec = self.embedder.embed_text(raw_response[:512])
        import numpy as np
        relevance = float(np.dot(q_vec, r_vec))
        relevance = max(0.0, min(1.0, round((relevance + 1.0) / 2.0, 3)))

        latency = int((time.perf_counter() - start_time) * 1000)

        return QueryResult(
            query=query_text,
            response=annotated_response,
            strategy=strategy,
            retrieved_chunks=candidates,
            claims=verifications,
            faithfulness_score=faithfulness,
            answer_relevance_score=relevance,
            latency_ms=latency,
            subgraph=subgraph,
        )

    async def query_stream(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronous generator yielding Server-Sent Events (SSE) for streaming responses.
        """
        start_time = time.perf_counter()
        strategy, _ = self.router.route(query_text)

        yield {"event": "routing", "data": {"strategy": strategy}}

        # Hybrid Retrieval
        candidates, subgraph = self.hybrid_retriever.retrieve(query=query_text, top_k=top_k)

        # Yield subgraph immediately so frontend canvas renders reasoning graph
        yield {
            "event": "subgraph",
            "data": {
                "nodes": subgraph.nodes,
                "edges": subgraph.edges,
            },
        }

        # Yield retrieved context candidates
        yield {
            "event": "sources",
            "data": [
                {
                    "chunk_id": c.chunk_id,
                    "document_title": c.document_title,
                    "heading_breadcrumbs": c.heading_breadcrumbs,
                    "score": c.score,
                    "content": c.content,
                }
                for c in candidates
            ],
        }

        # Token streaming
        full_tokens = []
        async for token in self.synthesizer.generate_stream(
            query=query_text,
            retrieved_contexts=candidates,
            subgraph=subgraph,
        ):
            full_tokens.append(token)
            yield {"event": "token", "data": {"token": token}}

        synthesized_text = "".join(full_tokens)

        # NLI Verification
        claims = self.claim_splitter.split_into_claims(synthesized_text)
        verifications, faithfulness = self.nli_verifier.verify_response(claims, candidates)

        yield {
            "event": "verification",
            "data": {
                "faithfulness_score": faithfulness,
                "claims": [v.model_dump(mode="json") for v in verifications],
            },
        }

        latency = int((time.perf_counter() - start_time) * 1000)
        yield {
            "event": "done",
            "data": {
                "latency_ms": latency,
                "total_claims": len(claims),
                "faithfulness_score": faithfulness,
            },
        }
