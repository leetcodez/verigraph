# Modern AI Architectures: GraphRAG and Verifiable Retrieval

## 1. Limitations of Naive Vector RAG
Standard Retrieval-Augmented Generation (RAG) relies on chunking text into fixed token blocks and performing cosine similarity searches over dense vector embeddings. While effective for simple semantic lookups, Naive RAG suffers from three critical failure modes:
1. Context Fragmentation: Relational ties between distant concepts are broken by rigid chunk boundaries.
2. Multi-Hop Blindspots: Cosine similarity identifies local text similarity but cannot traverse multi-hop entity dependency chains across documents.
3. Hallucination and Non-Attribution: Models synthesize fluent prose that mixes retrieved facts with fabricated assertions without strict citation grounding.

## 2. GraphRAG and Knowledge Graph Synthesis
GraphRAG integrates structured Knowledge Graphs with unstructured text embeddings.
- Entity & Relation Extraction: Documents are analyzed to extract canonical entities (nodes) and relational triples (edges).
- Subgraph Traversal: Queries expand into multi-hop neighborhood graphs, allowing the engine to reason across complex relationships like `(Subject) -> [PREVENTS] -> (Phenomenon)`.
- Topological Context Injection: Extracted subgraphs are serialized alongside textual chunks into the LLM context window.

## 3. Hybrid Retrieval and Reciprocal Rank Fusion (RRF)
Hybrid search overcomes vocabulary mismatch and out-of-domain vocabulary errors by combining:
- Dense Vector Search: Captures latent semantic intent and paraphrasing.
- Sparse BM25 Search: Guarantees exact matches for technical tokens, IDs, and domain-specific acronyms.
- Reciprocal Rank Fusion (RRF): Merges disparate score scales using rank reciprocals: `Score(d) = sum(1 / (k + rank(d)))` with standard constant `k=60`.

## 4. Real-Time NLI Claim Verification
To guarantee factual integrity, modern engines incorporate Natural Language Inference (NLI) verification guards.
- Atomic Proposition Splitting: The generated response is decomposed into individual verifiable claims.
- Entailment Scoring: Each claim is evaluated against retrieved evidence passages to classify its status as Entailed, Neutral, or Contradiction.
- Citation Mapping: Supported claims receive explicit inline citation references linking back to the exact chunk and graph node ID.
