"""简洁的日志初始化。"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None, json: bool = False) -> None:
    """初始化基础日志配置。

    - level：日志级别字符串。
    - log_dir：如指定则同时写入文件（轮转）。
    - json：是否采用 JSON 格式输出（占位，当前使用简洁文本）。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "runtime.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
        handlers.append(file_handler)

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt if not json else '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}',
        handlers=handlers,
        force=True,
    )
