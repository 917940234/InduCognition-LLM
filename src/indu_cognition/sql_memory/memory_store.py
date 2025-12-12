"""SQL 记忆库：存储 DDL/文档/参考 SQL 与互动经验，使用 Chroma。"""

from __future__ import annotations

from typing import List, Optional

from ..config.models import SQLMemoryConfig, VectorStoreConfig
from ..llm.providers.dashscope import DashScopeEmbeddingClient
from ..retrieval.vector_store import ChromaStore


class SQLMemoryStore:
    def __init__(self, cfg: SQLMemoryConfig, embedding_client: DashScopeEmbeddingClient) -> None:
        self.cfg = cfg
        self.embedding_client = embedding_client
        vec_cfg = VectorStoreConfig(
            persist_path=cfg.persist_path, collection=cfg.collection, embedding_model="text-embedding-v3"
        )
        self.store = ChromaStore(cfg=vec_cfg, embedding_client=embedding_client)

    def add_items(self, texts: List[str], ids: List[str], metadatas: Optional[List[dict]] = None) -> None:
        self.store.add_texts(texts=texts, ids=ids, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5) -> List[dict]:
        return self.store.similarity_search(query, k=k)
