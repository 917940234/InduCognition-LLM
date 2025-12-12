"""SQL 执行抽象（首选 MySQL）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class SQLExecutor:
    def __init__(self, dsn: str, execution_timeout_sec: int = 20, max_rows: int = 1000) -> None:
        self.engine: Engine = create_engine(dsn, pool_pre_ping=True)
        self.timeout = execution_timeout_sec
        self.max_rows = max_rows

    def run(self, sql: str) -> Dict[str, Any]:
        try:
            with self.engine.connect() as conn:
                result = conn.execution_options(timeout=self.timeout).execute(text(sql))
                rows = result.fetchmany(self.max_rows)
                cols = result.keys()
                return {"error": None, "columns": list(cols), "rows": [list(r) for r in rows]}
        except SQLAlchemyError as e:
            logger.warning("SQL 执行错误: %s", e)
            return {"error": str(e), "columns": [], "rows": []}
