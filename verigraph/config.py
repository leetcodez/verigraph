"""Configuration settings for VeriGraph."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VeriGraph global application settings."""

    # Server Settings
    app_name: str = "VeriGraph Knowledge Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8080

    # Paths & Storage
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    storage_dir: Path = data_dir / "storage"
    uploads_dir: Path = data_dir / "uploads"
    benchmarks_dir: Path = data_dir / "benchmarks"

    # Chunking Configuration
    chunk_size: int = 384  # Target tokens per chunk
    chunk_overlap: int = 64  # Overlapping tokens between adjacent chunks
    min_chunk_size: int = 40  # Avoid tiny orphan chunks

    # Embedding Configuration
    embedding_dim: int = 384
    embedding_model: str = "local-deterministic"  # local-deterministic, sentence-transformers, or openai

    # Retrieval & RRF Parameters
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    dense_top_k: int = 8
    bm25_top_k: int = 8
    graph_top_k: int = 8
    final_top_k: int = 5
    graph_max_hop_depth: int = 2

    # Verification Engine Settings
    nli_entailment_threshold: float = 0.65
    nli_contradiction_threshold: float = 0.55
    min_claim_words: int = 4

    # Optional Frontier LLM / External API keys
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def init_directories(self) -> None:
        """Ensure all storage and runtime directories exist."""
        for path in [self.data_dir, self.storage_dir, self.uploads_dir, self.benchmarks_dir]:
            path.mkdir(parents=True, exist_ok=True)


# Global singleton instance
settings = Settings()
settings.init_directories()
