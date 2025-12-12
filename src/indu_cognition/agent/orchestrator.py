"""Orchestrator 主循环（简化版）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from ..config.models import AgentConfig, AppConfig
from ..llm import ChatMessage, GEvalClient, LLMClients
from ..retrieval import BM25Store, ChromaStore, FeedbackUpdater, hybrid_search, make_hierarchical_chunks, synthesize_queries
from ..sql_memory import SQLExecutor, SQLMemoryStore, Text2SQLGenerator
from ..tools.simple import AVAILABLE_TOOLS
from .routing import route_task, select_tools
from .state import AgentState
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """串联检索 / Text-to-SQL / 工具（占位）的主循环。"""

    def __init__(
        self,
        app_config: AppConfig,
        llm_clients: LLMClients,
        vector_store: ChromaStore,
        bm25_store: BM25Store,
        feedback_updater: FeedbackUpdater,
        text2sql: Text2SQLGenerator,
    ) -> None:
        self.cfg = app_config
        self.clients = llm_clients
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.feedback_updater = feedback_updater
        self.text2sql = text2sql
        self.tool_registry = ToolRegistry(AVAILABLE_TOOLS)

    def parse_and_route(self, state: AgentState) -> str:
        route, scores = route_task(state.user_query, self.clients.chat, self.cfg.agent)
        logger.info("Route decision: %s (%s)", route, scores)
        return route

    def run_retrieval(self, state: AgentState) -> AgentState:
        results = hybrid_search(
            query=state.user_query,
            cfg=self.cfg.retrieval,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            rerank_client=self.clients.rerank,
        )
        state.contexts = [r.__dict__ for r in results]
        return state

    def run_sql(self, state: AgentState) -> AgentState:
        sql_res = self.text2sql.generate(state.user_query)
        state.sql_result = {"succeeded": sql_res.succeeded, "final_sql": sql_res.final_sql, "attempts": [a.__dict__ for a in sql_res.attempts]}
        return state

    def run_tools(self, state: AgentState) -> AgentState:
        tool_stats = [(s.name, s.success_rate) for s in self.tool_registry.get_stats()]
        selected = select_tools(state.user_query, tool_stats, self.cfg.agent, self.clients.chat)
        outputs: List[Dict[str, Any]] = []
        for name in selected:
            res = self.tool_registry.run_tool(name, state.user_query)
            outputs.append({"name": name, "success": res.success, "output": res.output, "metadata": res.metadata})
            if len(outputs) >= self.cfg.agent.max_tool_chain:
                break
        state.tool_outputs = outputs
        return state

    def synthesize(self, state: AgentState) -> AgentState:
        # 简化合成：将检索/SQL结果拼接交给 LLM 生成最终回答
        context_lines: List[str] = []
        for ctx in state.contexts:
            context_lines.append(f"[DOC]{ctx.get('text')}")
        if state.sql_result and state.sql_result.get("succeeded"):
            rows = state.sql_result.get("attempts", [])[-1].get("result", {}).get("rows", [])
            context_lines.append(f"[SQL_RESULT]{rows}")

        prompt = "请基于以下上下文回答用户问题，引用依据并保持简洁：\n" + "\n".join(context_lines)
        messages = [
            ChatMessage(role="system", content="你是钢包预热助手"),
            ChatMessage(role="user", content=f"问题：{state.user_query}\n{prompt}"),
        ]
        resp = self.clients.chat.generate(messages)
        state.response = resp.content
        return state

    def feedback(self, state: AgentState, user_feedback: Optional[str] = None) -> AgentState:
        if user_feedback in {"accepted", "corrected"} and state.response:
            qa_id = f"qa_{state.iteration}"
            self.feedback_updater.append_qa(qa_id=qa_id, question=state.user_query, answer=state.response)
        return state

    def run(self, query: str, user_feedback: Optional[str] = None) -> AgentState:
        state = AgentState(user_query=query)
        decision = self.parse_and_route(state)
        if decision == "sql":
            state = self.run_sql(state)
        elif decision == "tool":
            state = self.run_tools(state)
        else:
            state = self.run_retrieval(state)
        state = self.synthesize(state)
        state = self.feedback(state, user_feedback=user_feedback)
        return state


def build_orchestrator(app_config: AppConfig, llm_clients: LLMClients) -> Orchestrator:
    # 组装依赖
    vector_store = ChromaStore(app_config.vector_store, llm_clients.embedding)
    bm25_store = BM25Store(app_config.bm25)
    feedback_updater = FeedbackUpdater(
        retrieval_cfg=app_config.retrieval,
        vector_cfg=app_config.vector_store,
        embedding_client=llm_clients.embedding,
        bm25_store=bm25_store,
    )
    sql_memory = SQLMemoryStore(app_config.sql.memory_store, llm_clients.embedding)
    sql_executor = SQLExecutor(
        dsn=app_config.sql.dsn or "mysql+pymysql://user:pass@localhost:3306/indu_cognition",
        execution_timeout_sec=app_config.sql.execution_timeout_sec,
        max_rows=app_config.sql.max_rows,
    )
    text2sql = Text2SQLGenerator(app_config.sql, llm_clients.chat, sql_memory, sql_executor)
    return Orchestrator(
        app_config=app_config,
        llm_clients=llm_clients,
        vector_store=vector_store,
        bm25_store=bm25_store,
        feedback_updater=feedback_updater,
        text2sql=text2sql,
    )
