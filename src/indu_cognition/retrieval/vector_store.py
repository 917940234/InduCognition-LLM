"""Chroma 向量库封装。"""

from __future__ import annotations

from typing import List, Optional

import chromadb
from chromadb.api import ClientAPI

from ..config.models import VectorStoreConfig
from ..llm.providers.dashscope import DashScopeEmbeddingClient


class ChromaStore:
    def __init__(self, cfg: VectorStoreConfig, embedding_client: DashScopeEmbeddingClient) -> None:
        self.cfg = cfg
        self.embedding_client = embedding_client
        self.client: ClientAPI = chromadb.PersistentClient(path=cfg.persist_path)
        self.collection = self.client.get_or_create_collection(name=cfg.collection)

    def add_texts(self, texts: List[str], ids: List[str], metadatas: Optional[List[dict]] = None) -> None:
        embeddings = self.embedding_client.embed(texts).embeddings
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings)

    def similarity_search(self, query: str, k: int) -> List[dict]:
        emb = self.embedding_client.embed([query]).embeddings[0]
        res = self.collection.query(query_embeddings=[emb], n_results=k, include=["documents", "distances", "metadatas", "ids"])
        documents = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        results: List[dict] = []
        for doc, dist, meta, _id in zip(documents, distances, metadatas, ids):
            results.append({"id": _id, "text": doc, "distance": dist, "metadata": meta})
        return results
