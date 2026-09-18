"""Tests for SemanticChunker and hierarchy preservation."""

from verigraph.chunking.semantic_chunker import SemanticChunker


def test_semantic_chunker_basic():
    chunker = SemanticChunker(target_chunk_size=50, chunk_overlap=15, min_chunk_size=5)
    sections = [
        ("This is the first sentence. Here is the second sentence. Now comes the third one.", ["Heading 1"], 1)
    ]
    chunks = chunker.chunk_sections("doc1", "Doc Title", sections)
    assert len(chunks) >= 1
    assert chunks[0].document_title == "Doc Title"
    assert chunks[0].heading_breadcrumbs == ["Heading 1"]
    assert "first sentence" in chunks[0].content


def test_chunker_sliding_overlap():
    chunker = SemanticChunker(target_chunk_size=15, chunk_overlap=8, min_chunk_size=3)
    text = (
        "Alpha beta gamma delta epsilon. "
        "Zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron."
    )
    sections = [(text, ["Section A"], 1)]
    chunks = chunker.chunk_sections("doc2", "Alphabet", sections)
    assert len(chunks) >= 2
    # Check that chunks maintain sentence continuity
    assert any("Alpha" in c.content for c in chunks)
    assert any("Lambda" in c.content for c in chunks)
