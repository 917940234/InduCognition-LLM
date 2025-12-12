"""记忆增强 Text-to-SQL 迭代器。对应设计文档 Methods 3.3。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config.models import SQLConfig
from ..llm import ChatMessage
from ..llm.providers.deepseek import DeepSeekChatClient
from .executors import SQLExecutor
from .memory_store import SQLMemoryStore
from .prompt_builder import build_augmented_prompt


@dataclass
class SQLAttempt:
    sql: str
    result: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class Text2SQLResult:
    final_sql: Optional[str]
    attempts: List[SQLAttempt] = field(default_factory=list)
    succeeded: bool = False


class Text2SQLGenerator:
    def __init__(
        self,
        config: SQLConfig,
        chat_client: DeepSeekChatClient,
        memory_store: SQLMemoryStore,
        executor: SQLExecutor,
    ) -> None:
        self.cfg = config
        self.chat_client = chat_client
        self.memory_store = memory_store
        self.executor = executor

    def _retrieve_memory(self, question: str, k: int = 5) -> Dict[str, List[dict]]:
        hits = self.memory_store.similarity_search(question, k=k)
        # 简单分类：以 metadata.tag 作为分桶
        buckets: Dict[str, List[dict]] = {"ddl": [], "doc": [], "sql": [], "other": []}
        for h in hits:
            meta = h.get("metadata") or {}
            tag = meta.get("tag", "other")
            buckets.setdefault(tag, []).append({"text": h.get("text"), "metadata": meta})
        return buckets

    def _build_prompt(self, question: str, history_sql: List[str]) -> str:
        buckets = self._retrieve_memory(question)
        ddl_items = buckets.get("ddl", [])
        doc_items = buckets.get("doc", [])
        sql_items = buckets.get("sql", [])
        return build_augmented_prompt(question, ddl_items, doc_items, sql_items, history_sql)

    def generate(self, question: str, user_feedback: Optional[str] = None) -> Text2SQLResult:
        attempts: List[SQLAttempt] = []
        history_sql: List[str] = []
        for _ in range(self.cfg.t_max):
            prompt = self._build_prompt(question, history_sql)
            messages = [
                ChatMessage(role="system", content="你是 SQL 生成助手，请输出合法的 SQL。"),
                ChatMessage(role="user", content=prompt),
            ]
            resp = self.chat_client.generate(messages)
            sql_stmt = resp.content.strip()
            history_sql.append(sql_stmt)
            exec_result = self.executor.run(sql_stmt)
            attempt = SQLAttempt(sql=sql_stmt, result=exec_result, error=exec_result.get("error"))
            attempts.append(attempt)
            if not exec_result.get("error"):
                return Text2SQLResult(final_sql=sql_stmt, attempts=attempts, succeeded=True)
        # 达到上限，失败
        return Text2SQLResult(final_sql=None, attempts=attempts, succeeded=False)
