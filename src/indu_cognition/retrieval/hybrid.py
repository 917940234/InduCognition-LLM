"""混合检索：向量 + BM25 + 自适应 Top-K + Rerank。"""

from __future__ import annotations

import math
import statistics
from typing import List

import numpy as np

from ..config.models import RetrievalConfig
from ..llm.providers.dashscope import QwenRerankClient
from .bm25_store import BM25Store
from .types import RetrievalCandidate, RetrievalResult
from .vector_store import ChromaStore


def _cosine_from_distance(distance: float) -> float:
    # Chroma 距离默认为 L2/内积，这里简单映射，非运行时精确，仅用于排序占位。
    return 1.0 - distance


def hybrid_search(
    query: str,
    cfg: RetrievalConfig,
    vector_store: ChromaStore,
    bm25_store: BM25Store,
    rerank_client: QwenRerankClient,
) -> List[RetrievalResult]:
    k = max(1, math.floor(cfg.llm_context_len / max(cfg.expected_chunk_len, 1)))

    vector_hits = vector_store.similarity_search(query, k=cfg.top_k)
    vector_candidates: List[RetrievalCandidate] = []
    for hit in vector_hits:
        score_cos = _cosine_from_distance(hit.get("distance", 0.0))
        meta = hit.get("metadata", {}) or {}
        vector_candidates.append(
            RetrievalCandidate(
                text=hit["text"],
                source_id=hit.get("id", ""),
                parent_id=meta.get("parent_id"),
                score_cosine=score_cos,
                metadata=meta,
            )
        )

    bm25_hits = bm25_store.search(query, k=cfg.top_k)
    bm25_candidates: List[RetrievalCandidate] = []
    for text, score_bm25, meta in bm25_hits:
        bm25_candidates.append(
            RetrievalCandidate(
                text=text,
                source_id=meta.get("id", ""),
                parent_id=meta.get("parent_id"),
                score_bm25=score_bm25,
                metadata=meta,
            )
        )

    merged = {}
    for cand in vector_candidates + bm25_candidates:
        key = cand.source_id or cand.text
        if key not in merged:
            merged[key] = cand
        else:
            merged[key].score_cosine = max(merged[key].score_cosine, cand.score_cosine)
            merged[key].score_bm25 = max(merged[key].score_bm25, cand.score_bm25)

    candidates = list(merged.values())
    for cand in candidates:
        cand.score_hybrid = cfg.lambda_hybrid * cand.score_cosine + (1 - cfg.lambda_hybrid) * cand.score_bm25

    hybrid_scores = [c.score_hybrid for c in candidates]
    if hybrid_scores:
        mu = statistics.mean(hybrid_scores)
        sigma = statistics.pstdev(hybrid_scores) if len(hybrid_scores) > 1 else 0.0
    else:
        mu = 0.0
        sigma = 0.0
    tau = mu + cfg.gamma_filter * sigma

    filtered = [c for c in candidates if c.score_hybrid > tau]
    if not filtered:
        filtered = sorted(candidates, key=lambda x: x.score_hybrid, reverse=True)[: cfg.top_k]

    documents = [c.text for c in filtered]
    rerank_resp = rerank_client.rerank(query=query, documents=documents, top_n=min(len(documents), k))
    rerank_scores = {res.text: res.score for res in rerank_resp.results}

    results: List[RetrievalResult] = []
    for cand in filtered:
        results.append(
            RetrievalResult(
                text=cand.text,
                source_id=cand.source_id,
                parent_id=cand.parent_id,
                score=rerank_scores.get(cand.text, cand.score_hybrid),
                metadata=cand.metadata,
                raw={"hybrid": cand.score_hybrid},
            )
        )

    return sorted(results, key=lambda x: x.score, reverse=True)[: k]
