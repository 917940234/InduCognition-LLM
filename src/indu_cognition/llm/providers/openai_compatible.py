"""兼容 OpenAI API 的轻量客户端。

用于 DeepSeek Chat、DashScope embedding/rerank 等 openai-compatible 场景。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from ..types.base import ChatMessage, EmbeddingResponse, LLMResponse, RerankItem, RerankResponse

logger = logging.getLogger(__name__)


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        default_params: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.default_params = default_params or {}

    def generate(self, messages: List[ChatMessage], **kwargs: Any) -> LLMResponse:
        if not self.api_key:
            raise ValueError("API key 未配置，无法调用 chat 接口")
        payload = {
            "model": self.model,
            "messages": [m.__dict__ for m in messages],
            **self.default_params,
            **kwargs,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage")
        return LLMResponse(content=content, raw=data, usage=usage)


class OpenAIEmbeddingClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str], timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: List[str]) -> EmbeddingResponse:
        if not self.api_key:
            raise ValueError("API key 未配置，无法调用 embedding 接口")
        payload = {"model": self.model, "input": texts}
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        return EmbeddingResponse(embeddings=embeddings, raw=data)


class OpenAIRerankClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        default_top_n: int = 50,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.default_top_n = default_top_n
        self.timeout = timeout

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        if not self.api_key:
            raise ValueError("API key 未配置，无法调用 rerank 接口")
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n or self.default_top_n,
        }
        response = requests.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        results: List[RerankItem] = []
        for idx, item in enumerate(data.get("results", [])):
            results.append(
                RerankItem(
                    index=item.get("index", idx),
                    score=item.get("relevance_score", 0.0),
                    text=item.get("document", ""),
                )
            )
        return RerankResponse(results=results, raw=data)
