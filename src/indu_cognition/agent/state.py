"""Agent 状态定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    user_query: str
    iteration: int = 0
    depth: int = 0
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    sql_result: Optional[Dict[str, Any]] = None
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    response: Optional[str] = None
