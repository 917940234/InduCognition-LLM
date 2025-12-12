"""DashScope 兼容模型封装（embedding / rerank）。"""

from __future__ import annotations

from typing import List, Optional

from ...config.models import EmbeddingConfig, RerankConfig
from ..types.base import EmbeddingResponse, RerankResponse
from .openai_compatible import OpenAIEmbeddingClient, OpenAIRerankClient


class DashScopeEmbeddingClient:
    """使用 text-embedding-v3 的封装。"""

    def __init__(self, config: EmbeddingConfig, timeout: int = 60) -> None:
        self.client = OpenAIEmbeddingClient(
            base_url=config.base_url, model=config.model, api_key=config.api_key, timeout=timeout
        )

    def embed(self, texts: List[str]) -> EmbeddingResponse:
        return self.client.embed(texts)


class QwenRerankClient:
    """Qwen rerank 封装。"""

    def __init__(self, config: RerankConfig, timeout: int = 60) -> None:
        self.client = OpenAIRerankClient(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            default_top_n=config.top_n,
            timeout=timeout,
        )

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        return self.client.rerank(query=query, documents=documents, top_n=top_n)
