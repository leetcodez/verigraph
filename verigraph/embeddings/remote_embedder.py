"""Remote embedding provider with seamless local fallback."""

from typing import List, Optional
import httpx
from verigraph.core.logging import logger
from verigraph.embeddings.base import BaseEmbedder
from verigraph.embeddings.local_embedder import LocalSemanticEmbedder


class RemoteEmbedder(BaseEmbedder):
    """
    Calls OpenAI-compatible / Ollama embedding endpoints.
    Falls back to LocalSemanticEmbedder if offline or API error occurs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        dimension: int = 384,
    ):
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._model_name = model_name
        self._dim = dimension
        self._fallback = LocalSemanticEmbedder(dimension=dimension)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        res = self.embed_batch([text])
        return res[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._api_key and "localhost" not in self._base_url and "127.0.0.1" not in self._base_url:
            return self._fallback.embed_batch(texts)

        try:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            endpoint = f"{self._base_url}/embeddings"
            payload = {"input": texts, "model": self._model_name}

            with httpx.Client(timeout=10.0) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    return embeddings
                else:
                    logger.warning(f"Remote embedding failed with status {resp.status_code}, falling back to local embedder")
        except Exception as e:
            logger.warning(f"Remote embedding exception: {e}, falling back to local embedder")

        return self._fallback.embed_batch(texts)
