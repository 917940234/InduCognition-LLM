"""路由与工具选择策略。对应设计文档 Methods 3.1。"""

from __future__ import annotations

from typing import List, Tuple

from ..config.models import AgentConfig
from ..llm import ChatMessage
from ..llm.providers.deepseek import DeepSeekChatClient


def route_task(user_query: str, chat_client: DeepSeekChatClient, cfg: AgentConfig) -> Tuple[str, dict]:
    """LLM + 阈值路由，返回类别与原始打分。"""
    prompt = (
        "对用户问题进行类型路由，返回 JSON："
        '{"retrieval":prob_retrieval,"sql":prob_sql,"tool":prob_tool}.'
        "检索=文档/知识库，sql=数据库/查询，tool=其他计算/分析。"
    )
    messages = [ChatMessage(role="system", content="你是路由器"), ChatMessage(role="user", content=f"{prompt}\n问题: {user_query}")]
    resp = chat_client.generate(messages).content
    try:
        import json

        data = json.loads(resp)
        candidates = {
            "retrieval": float(data.get("retrieval", 0)),
            "sql": float(data.get("sql", 0)),
            "tool": float(data.get("tool", 0)),
        }
    except Exception:
        candidates = {"retrieval": 1.0, "sql": 0.0, "tool": 0.0}

    best = max(candidates.items(), key=lambda x: x[1])
    decision = best[0] if best[1] >= cfg.routing_threshold else "retrieval"
    return decision, candidates


def select_tools(user_query: str, tool_stats: List[Tuple[str, float]], cfg: AgentConfig, chat_client: DeepSeekChatClient) -> List[str]:
    """根据 LLM 预测 + 历史成功率选择工具。"""
    # 使用 LLM 评分工具适配度
    tool_names = [name for name, _ in tool_stats]
    prompt = (
        "针对问题选择合适的工具，并为每个工具给出0-1的置信度，返回JSON对象，key为工具名，value为概率。"
        f"工具列表: {tool_names}"
    )
    messages = [ChatMessage(role="system", content="你是工具选择器"), ChatMessage(role="user", content=f"{prompt}\n问题: {user_query}")]
    llm_scores: dict
    try:
        import json

        resp = chat_client.generate(messages).content
        llm_scores = json.loads(resp)
    except Exception:
        llm_scores = {name: 0.5 for name in tool_names}

    selected = []
    for name, hist_succ in tool_stats:
        p_llm = float(llm_scores.get(name, 0.0))
        score = cfg.alpha_tool * p_llm + (1 - cfg.alpha_tool) * hist_succ
        if score >= cfg.tool_utility_threshold:
            selected.append(name)
    return selected[: cfg.max_tool_chain]
