"""BM25 索引封装（使用 rank_bm25）。"""

from __future__ import annotations

from typing import List, Tuple

from rank_bm25 import BM25Okapi

from ..config.models import BM25Config


class BM25Store:
    def __init__(self, cfg: BM25Config) -> None:
        self.cfg = cfg
        self.corpus: List[List[str]] = []
        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        self.ids: List[str] = []
        self.model: BM25Okapi | None = None

    def add(self, queries: List[str], ids: List[str], metadatas: List[dict] | None = None) -> None:
        metadatas = metadatas or [{} for _ in queries]
        tokenized = [q.split() for q in queries]
        self.corpus.extend(tokenized)
        self.texts.extend(queries)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        self.model = BM25Okapi(self.corpus, k1=self.cfg.k1, b=self.cfg.b)

    def search(self, query: str, k: int) -> List[Tuple[str, float, dict]]:
        if not self.model:
            return []
        scores = self.model.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results: List[Tuple[str, float, dict]] = []
        for idx, score in ranked:
            results.append((self.texts[idx], float(score), {"id": self.ids[idx], **self.metadatas[idx]}))
        return results
