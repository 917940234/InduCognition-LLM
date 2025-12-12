"""LLM 客户端集合结构。"""

from __future__ import annotations

from dataclasses import dataclass

from .base import EmbeddingResponse, LLMResponse, RerankResponse


@dataclass
class ChatClientProtocol:
    def generate(self, messages: list, **kwargs) -> LLMResponse:  # pragma: no cover - 协议占位
        ...


@dataclass
class EmbeddingClientProtocol:
    def embed(self, texts: list[str]) -> EmbeddingResponse:  # pragma: no cover - 协议占位
        ...


@dataclass
class RerankClientProtocol:
    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> RerankResponse:  # pragma: no cover - 协议占位
        ...


@dataclass
class LLMClients:
    chat: ChatClientProtocol
    embedding: EmbeddingClientProtocol
    rerank: RerankClientProtocol
