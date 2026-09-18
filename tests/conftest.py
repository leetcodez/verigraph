"""Pytest fixtures and configuration."""

import shutil
import tempfile
from pathlib import Path
import pytest
from verigraph.config import Settings
from verigraph.core.models import Chunk, Document
from verigraph.engine.pipeline import VeriGraphPipeline


@pytest.fixture
def temp_env(monkeypatch):
    """Creates a temporary test sandbox directory."""
    temp_dir = Path(tempfile.mkdtemp())
    test_settings = Settings(
        base_dir=temp_dir,
        data_dir=temp_dir / "data",
        storage_dir=temp_dir / "storage",
        uploads_dir=temp_dir / "uploads",
        benchmarks_dir=temp_dir / "benchmarks",
        chunk_size=120,
        chunk_overlap=25,
        min_chunk_size=10,
    )
    test_settings.init_directories()

    # Monkeypatch settings in modules
    import verigraph.config
    monkeypatch.setattr(verigraph.config, "settings", test_settings)

    yield test_settings
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pipeline(temp_env):
    """Provides an initialized VeriGraph pipeline inside a clean sandbox."""
    pipeline = VeriGraphPipeline()
    return pipeline


@pytest.fixture
def sample_chunks():
    """Provides sample chunks for retrieval tests."""
    return [
        Chunk(
            document_id="doc-raft",
            document_title="Raft Protocol",
            chunk_index=0,
            content="Raft achieves consensus by electing a distinguished leader. The leader manages log replication.",
            token_count=18,
            heading_breadcrumbs=["Distributed Systems", "Raft", "Leader Election"],
        ),
        Chunk(
            document_id="doc-raft",
            document_title="Raft Protocol",
            chunk_index=1,
            content="To prevent split-brain during network partitions, Raft requires a majority quorum for elections.",
            token_count=18,
            heading_breadcrumbs=["Distributed Systems", "Raft", "Fault Tolerance"],
        ),
        Chunk(
            document_id="doc-security",
            document_title="Zero Trust Security",
            chunk_index=0,
            content="Mutual TLS (mTLS) provides end-to-end cryptographic transport security and service authentication.",
            token_count=16,
            heading_breadcrumbs=["Cloud Security", "mTLS"],
        ),
    ]
