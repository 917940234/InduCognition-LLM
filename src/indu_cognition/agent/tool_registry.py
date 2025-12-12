"""工具注册与历史成功率记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..tools.base import BaseTool, ToolResult


@dataclass
class ToolStats:
    name: str
    success_count: int = 0
    total_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.5  # 初始保守值
        return self.success_count / self.total_count


class ToolRegistry:
    def __init__(self, tools: List[BaseTool]) -> None:
        self.tools = {t.name: t for t in tools}
        self.stats: Dict[str, ToolStats] = {name: ToolStats(name=name) for name in self.tools}

    def list_tools(self) -> List[BaseTool]:
        return list(self.tools.values())

    def get_stats(self) -> List[ToolStats]:
        return list(self.stats.values())

    def run_tool(self, name: str, query: str) -> ToolResult:
        tool = self.tools[name]
        res = tool.run(query)
        stat = self.stats[name]
        stat.total_count += 1
        if res.success:
            stat.success_count += 1
        return res
