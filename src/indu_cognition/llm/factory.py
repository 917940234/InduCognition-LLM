"""LLM 客户端工厂。"""

from __future__ import annotations

from .providers.dashscope import DashScopeEmbeddingClient, QwenRerankClient
from .providers.deepseek import DeepSeekChatClient
from .types.clients import LLMClients
from ..config.models import AppConfig


def build_llm_clients(config: AppConfig) -> LLMClients:
    """根据全局配置实例化 chat / embedding / rerank 客户端。"""
    chat = DeepSeekChatClient(config.llm)
    embedding = DashScopeEmbeddingClient(config.embedding)
    rerank = QwenRerankClient(config.rerank)
    return LLMClients(chat=chat, embedding=embedding, rerank=rerank)
