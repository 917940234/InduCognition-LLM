"""反馈驱动的增量更新。"""

from __future__ import annotations

from typing import List

from ..config.models import RetrievalConfig, VectorStoreConfig
from ..llm.providers.dashscope import DashScopeEmbeddingClient
from .bm25_store import BM25Store
from .chunking import make_hierarchical_chunks
from .vector_store import ChromaStore


class FeedbackUpdater:
    """接受 Q&A 反馈，追加到向量库与 BM25 索引。"""

    def __init__(
        self,
        retrieval_cfg: RetrievalConfig,
        vector_cfg: VectorStoreConfig,
        embedding_client: DashScopeEmbeddingClient,
        bm25_store: BM25Store,
    ) -> None:
        self.retrieval_cfg = retrieval_cfg
        self.vector_store = ChromaStore(vector_cfg, embedding_client)
        self.bm25_store = bm25_store

    def append_qa(self, qa_id: str, question: str, answer: str) -> None:
        text = f"Q: {question}\nA: {answer}"
        chunks = make_hierarchical_chunks(qa_id, text, self.retrieval_cfg)
        child_chunks = [c for c in chunks if c.level == "child"]
        docs = [c.text for c in child_chunks]
        ids = [c.chunk_id for c in child_chunks]
        metas = [{"parent_id": c.parent_id, "qa_id": qa_id, "level": c.level} for c in child_chunks]
        self.vector_store.add_texts(docs, ids, metas)

        # BM25 使用合成查询近似文本自身
        self.bm25_store.add(queries=docs, ids=ids, metadatas=metas)
