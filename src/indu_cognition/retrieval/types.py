from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RetrievalCandidate:
    text: str
    source_id: str
    parent_id: Optional[str]
    score_cosine: float = 0.0
    score_bm25: float = 0.0
    score_hybrid: float = 0.0
    metadata: Optional[dict] = None


@dataclass
class RetrievalResult:
    text: str
    source_id: str
    parent_id: Optional[str]
    score: float
    metadata: Optional[dict] = None
    raw: Optional[Any] = None
