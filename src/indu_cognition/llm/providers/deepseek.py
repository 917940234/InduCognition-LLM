"""DeepSeek Chat 客户端封装。"""

from __future__ import annotations

from typing import Any, List

from ...config.models import LLMConfig
from ..types.base import ChatMessage, LLMResponse
from .openai_compatible import OpenAIChatClient


class DeepSeekChatClient:
    """针对 deepseek-chat 的轻量包装。"""

    def __init__(self, config: LLMConfig) -> None:
        default_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "frequency_penalty": config.frequency_penalty,
        }
        self.client = OpenAIChatClient(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            default_params=default_params,
        )

    def generate(self, messages: List[ChatMessage], **kwargs: Any) -> LLMResponse:
        return self.client.generate(messages, **kwargs)
