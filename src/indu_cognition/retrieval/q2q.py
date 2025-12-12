"""合成查询（Q2Q）。对应设计文档 Methods 3.2 Q2Q 索引。"""

from __future__ import annotations

from typing import List

from ..config import RetrievalConfig
from ..llm import ChatMessage
from ..llm.providers.deepseek import DeepSeekChatClient


def synthesize_queries(chunks: List[str], chat_client: DeepSeekChatClient, prompt_template: str | None = None) -> List[str]:
    """为每个文本块生成合成查询。

    prompt_template 如未提供，使用默认模板，要求返回单条查询。
    """
    synthesized: List[str] = []
    for chunk in chunks:
        system_prompt = "你是工业文档索引助手，请为给定文本生成最可能的用户查询。只返回一句查询。"
        user_prompt = prompt_template or "文本：{text}\n请给出一个最可能的用户查询。"
        content = user_prompt.format(text=chunk)
        messages = [ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=content)]
        resp = chat_client.generate(messages)
        synthesized.append(resp.content.strip())
    return synthesized
