"""工具接口与结果类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolResult:
    success: bool
    output: Any
    metadata: Dict[str, Any]


class BaseTool:
    name: str = "base"
    description: str = "abstract tool"

    def run(self, query: str) -> ToolResult:  # pragma: no cover - 接口占位
        raise NotImplementedError
