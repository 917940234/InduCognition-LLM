"""简单工具实现（本地轻量计算）。"""

from __future__ import annotations

import re
from typing import List

from .base import BaseTool, ToolResult


class KeywordExtractTool(BaseTool):
    name = "keyword_extract"
    description = "提取查询中的关键词（简单基于词频的启发式）。"

    def run(self, query: str) -> ToolResult:
        tokens = re.findall(r"\w+", query.lower())
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [k for k, _ in top]
        return ToolResult(success=True, output={"keywords": keywords}, metadata={"count": len(keywords)})


class NumericSummaryTool(BaseTool):
    name = "numeric_summary"
    description = "从查询文本中提取数字并给出均值/最大/最小。"

    def run(self, query: str) -> ToolResult:
        nums: List[float] = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", query)]
        if not nums:
            return ToolResult(success=False, output="未找到数字", metadata={})
        summary = {"count": len(nums), "min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums)}
        return ToolResult(success=True, output=summary, metadata={"numbers": nums})


AVAILABLE_TOOLS = [KeywordExtractTool(), NumericSummaryTool()]
