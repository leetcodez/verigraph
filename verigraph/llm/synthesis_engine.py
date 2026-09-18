"""Multi-hop synthesis engine supporting OpenAI, Ollama, and local deterministic synthesis."""

import asyncio
import re
from typing import AsyncGenerator, Dict, List, Optional
import httpx
from verigraph.config import settings
from verigraph.core.logging import logger
from verigraph.core.models import RetrievalCandidate, Subgraph
from verigraph.llm.base import BaseSynthesizer


class SynthesisEngine(BaseSynthesizer):
    """
    Synthesizes grounded answers from retrieved context chunks and knowledge graph subgraphs.
    Supports Frontier LLMs (OpenAI, Groq, Anthropic-compatible) and local deterministic synthesis.
    """

    SYSTEM_PROMPT = (
        "You are VeriGraph, an enterprise-grade verifiable knowledge engine. "
        "Your task is to answer the user's query strictly using the provided context chunks "
        "and knowledge graph relationships. Every factual assertion must be faithful to the source. "
        "If the context does not contain sufficient facts to answer, explicitly state what is missing."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        self.model_name = model_name or settings.openai_model

    def _build_context_prompt(
        self,
        query: str,
        retrieved_contexts: List[RetrievalCandidate],
        subgraph: Optional[Subgraph] = None,
    ) -> str:
        """Constructs a structured prompt containing graph relations and chunk passages."""
        parts = [f"User Query: {query}\n\n=== VERIFIED SOURCE CONTEXT ==="]

        if subgraph and subgraph.edges:
            parts.append("\n[Knowledge Graph Relationships]:")
            for edge in subgraph.edges[:12]:
                parts.append(
                    f"• ({edge.get('source')}) -[{edge.get('relation')}]-> ({edge.get('target')})"
                )

        if retrieved_contexts:
            parts.append("\n[Document Excerpts]:")
            for idx, cand in enumerate(retrieved_contexts, start=1):
                breadcrumbs = " > ".join(cand.heading_breadcrumbs) if cand.heading_breadcrumbs else "Root"
                parts.append(
                    f"\n[Source #{idx}: {cand.document_title} | Section: {breadcrumbs}]:\n"
                    f"{cand.content.strip()}"
                )

        parts.append(
            "\n\n=== INSTRUCTIONS ===\n"
            "Synthesize a clear, direct, and factual answer based ONLY on the evidence above. "
            "Organize your thoughts into distinct declarative statements so they can be verified."
        )
        return "\n".join(parts)

    def _deterministic_synthesis(
        self,
        query: str,
        retrieved_contexts: List[RetrievalCandidate],
        subgraph: Optional[Subgraph] = None,
    ) -> str:
        """
        High-fidelity deterministic local synthesizer.
        Extracts salient statements, joins multi-hop relationships, and forms a coherent answer.
        """
        if not retrieved_contexts:
            return (
                f"No verified information regarding '{query}' was found in the indexed corpus. "
                "Please ingest relevant documents to query this topic."
            )

        statements: List[str] = []
        seen_sentences = set()

        # 1. Synthesize knowledge graph relational facts
        if subgraph and subgraph.edges:
            rel_summaries = []
            for edge in subgraph.edges[:5]:
                src = edge.get("source", "").replace("_", " ").title()
                rel = edge.get("relation", "").replace("_", " ").lower()
                tgt = edge.get("target", "").replace("_", " ").title()
                rel_summaries.append(f"{src} {rel} {tgt}")

            if rel_summaries:
                statements.append(
                    f"According to the knowledge graph, {', while '.join(rel_summaries)}."
                )

        # 2. Extract salient sentences from top retrieved chunks
        query_words = {w.lower() for w in re.findall(r"\b\w+\b", query) if len(w) > 2}

        for cand in retrieved_contexts[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", cand.content)
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 25 or sent_clean in seen_sentences:
                    continue

                sent_words = {w.lower() for w in re.findall(r"\b\w+\b", sent_clean)}
                overlap = len(query_words & sent_words)
                if overlap >= 1 or len(statements) < 3:
                    seen_sentences.add(sent_clean)
                    statements.append(sent_clean)
                    if len(statements) >= 5:
                        break

        if not statements:
            # Fallback to direct chunk content
            statements.append(retrieved_contexts[0].content[:400].strip() + "...")

        return " ".join(statements)

    def generate(
        self,
        query: str,
        retrieved_contexts: List[RetrievalCandidate],
        subgraph: Optional[Subgraph] = None,
    ) -> str:
        # Check if remote LLM configured
        if self.api_key:
            try:
                prompt = self._build_context_prompt(query, retrieved_contexts, subgraph)
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"Remote LLM call failed: {e}, falling back to deterministic synthesis")

        # Fallback to local deterministic synthesis
        return self._deterministic_synthesis(query, retrieved_contexts, subgraph)

    async def generate_stream(
        self,
        query: str,
        retrieved_contexts: List[RetrievalCandidate],
        subgraph: Optional[Subgraph] = None,
    ) -> AsyncGenerator[str, None]:
        # If remote API with streaming
        if self.api_key:
            prompt = self._build_context_prompt(query, retrieved_contexts, subgraph)
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "stream": True,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            try:
                async with httpx.AsyncClient(timeout=40.0) as client:
                    async with client.stream(
                        "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.startswith("data: ") and not line.endswith("[DONE]"):
                                    try:
                                        import json
                                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                                        if delta:
                                            yield delta
                                    except Exception:
                                        continue
                            return
            except Exception as e:
                logger.warning(f"Remote LLM streaming failed: {e}, using local streaming")

        # Local deterministic streaming simulation
        text = self._deterministic_synthesis(query, retrieved_contexts, subgraph)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)
