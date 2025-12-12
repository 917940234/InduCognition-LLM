"""基础类型定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    raw: Dict[str, Any]


@dataclass
class RerankItem:
    index: int
    score: float
    text: str


@dataclass
class RerankResponse:
    results: List[RerankItem]
    raw: Dict[str, Any]
