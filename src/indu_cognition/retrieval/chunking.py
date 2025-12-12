"""文档切分与层次化块结构。

对应设计文档 Methods 3.2：父/子块切分 + 重叠。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..config import RetrievalConfig


@dataclass
class Chunk:
    text: str
    parent_id: str
    chunk_id: str
    level: str  # "parent" or "child"


def _split_by_delimiters(text: str, delimiters: Iterable[str]) -> List[str]:
    """按多个分隔符顺序切分，保持顺序拼接。"""
    parts = [text]
    for delim in delimiters:
        next_parts: List[str] = []
        for p in parts:
            next_parts.extend(p.split(delim))
        parts = next_parts
    # 过滤空白
    return [p.strip() for p in parts if p.strip()]


def split_children(text: str, cfg: RetrievalConfig) -> List[str]:
    """按子块大小与重叠近似切分。

    采用字符长度近似 token，重叠通过窗口滑动实现。
    """
    raw_segments = _split_by_delimiters(text, cfg.child_delimiters)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for seg in raw_segments:
        seg_len = len(seg)
        if current and current_len + seg_len > cfg.child_max_tokens:
            chunks.append(" ".join(current).strip())
            # 重叠：保留末尾重叠近似
            overlap_chars = max(cfg.overlap, 0)
            if overlap_chars > 0 and chunks[-1]:
                tail = chunks[-1][-overlap_chars:]
                current = [tail]
                current_len = len(tail)
            else:
                current = []
                current_len = 0
        current.append(seg)
        current_len += seg_len
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def make_hierarchical_chunks(doc_id: str, text: str, cfg: RetrievalConfig) -> List[Chunk]:
    """生成父/子块层次结构."""
    children = split_children(text, cfg)
    chunks: List[Chunk] = []
    parent_buffer: List[str] = []
    parent_len = 0
    parent_idx = 0
    child_idx = 0
    for child in children:
        if parent_buffer and parent_len + len(child) > cfg.parent_max_tokens:
            parent_text = " ".join(parent_buffer).strip()
            parent_id = f"{doc_id}_p{parent_idx}"
            chunks.append(Chunk(text=parent_text, parent_id=parent_id, chunk_id=parent_id, level="parent"))
            parent_idx += 1
            parent_buffer = []
            parent_len = 0
        parent_buffer.append(child)
        parent_len += len(child)
        parent_id = f"{doc_id}_p{parent_idx}"
        chunk_id = f"{doc_id}_c{child_idx}"
        chunks.append(Chunk(text=child, parent_id=parent_id, chunk_id=chunk_id, level="child"))
        child_idx += 1

    if parent_buffer:
        parent_text = " ".join(parent_buffer).strip()
        parent_id = f"{doc_id}_p{parent_idx}"
        chunks.append(Chunk(text=parent_text, parent_id=parent_id, chunk_id=parent_id, level="parent"))
    return chunks
