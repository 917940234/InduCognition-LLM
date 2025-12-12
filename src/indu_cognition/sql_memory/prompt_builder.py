"""构建 Text-to-SQL 增强 prompt。"""

from __future__ import annotations

from typing import List


def format_memory_items(items: List[dict], header: str) -> str:
    lines = [header]
    for it in items:
        text = it.get("text") or it.get("document") or ""
        meta = it.get("metadata") or {}
        tag = meta.get("tag", "memory")
        lines.append(f"- [{tag}] {text}")
    return "\n".join(lines)


def build_augmented_prompt(
    question: str,
    ddl_items: List[dict],
    doc_items: List[dict],
    sql_examples: List[dict],
    history_sql: List[str] | None = None,
) -> str:
    segments = [
        "你是一名工业场景的 SQL 助手，请基于提供的元数据生成安全、正确的 SQL。",
        format_memory_items(ddl_items, "数据库 DDL："),
        format_memory_items(doc_items, "相关文档片段："),
        format_memory_items(sql_examples, "参考 SQL/模板："),
        f"用户问题：{question}",
    ]
    if history_sql:
        segments.append("之前尝试的 SQL：")
        segments.extend([f"- {s}" for s in history_sql])
    segments.append("请仅输出 SQL 语句，不要解释。")
    return "\n\n".join([s for s in segments if s])
