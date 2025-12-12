"""G-Eval 风格的评估客户端占位实现。"""

from __future__ import annotations

from typing import Dict

from ...config.models import EvaluationConfig
from ..types.base import ChatMessage, LLMResponse
from ..providers.deepseek import DeepSeekChatClient


class GEvalClient:
    """使用 DeepSeek Chat 按 Applause rubric 做回答质量评估的轻量封装。"""

    def __init__(self, chat_client: DeepSeekChatClient, config: EvaluationConfig) -> None:
        self.chat_client = chat_client
        self.config = config

    def _build_prompt(self, question: str, reference: str, answer: str) -> str:
        rubric = self.config.applause_rubric
        lines = [
            "请根据以下维度为模型回答打分（1-5 分，整数），并给出每个维度的简要理由：",
            f"- Accuracy ({rubric.get('accuracy', 0)}): 与参考答案/事实一致性",
            f"- Relevance ({rubric.get('relevance', 0)}): 与问题相关度",
            f"- Completeness ({rubric.get('completeness', 0)}): 关键要点覆盖",
            f"- Clarity ({rubric.get('clarity', 0)}): 叙述清晰度",
            f"- Language & Grammar ({rubric.get('language_grammar', 0)}): 语言与语法",
            "",
            "请返回 JSON：{\"accuracy\":n1,\"relevance\":n2,\"completeness\":n3,\"clarity\":n4,\"language_grammar\":n5,\"comment\":\"...\"}",
            "",
            f"问题: {question}",
            f"参考答案: {reference}",
            f"模型答案: {answer}",
        ]
        return "\n".join(lines)

    def score(self, question: str, reference: str, answer: str) -> LLMResponse:
        prompt = self._build_prompt(question, reference, answer)
        messages = [
            ChatMessage(role="system", content="你是严谨的评估员，请仅返回 JSON 评分。"),
            ChatMessage(role="user", content=prompt),
        ]
        return self.chat_client.generate(messages)
