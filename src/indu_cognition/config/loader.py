"""配置加载与合并工具。

支持：
- 默认 YAML 配置；
- providers YAML（覆盖模型与秘钥字段）；
- .env 环境变量（用于秘钥/DSN 等敏感信息）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from .models import AppConfig


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """浅显的递归合并，右侧覆盖左侧。"""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_app_config(
    config_path: Path,
    providers_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
) -> AppConfig:
    """加载应用配置，合并 providers 与环境变量覆盖。"""
    if env_path:
        load_dotenv(env_path, override=False)

    config_data = _read_yaml(config_path)
    providers_data = _read_yaml(providers_path) if providers_path else {}

    merged = _deep_update(config_data, providers_data)

    # 环境变量覆盖常用敏感项
    env_overrides = {
        "llm": {"api_key": os.getenv("DEEPSEEK_API_KEY")},
        "embedding": {"api_key": os.getenv("DASHSCOPE_API_KEY")},
        "rerank": {"api_key": os.getenv("DASHSCOPE_API_KEY")},
        "evaluation": {"g_eval_api_key": os.getenv("DEEPSEEK_API_KEY")},
        "sql": {"dsn": os.getenv("MYSQL_DSN")},
    }
    env_overrides = {
        section: {k: v for k, v in values.items() if v}
        for section, values in env_overrides.items()
    }
    merged = _deep_update(merged, env_overrides)

    return AppConfig.model_validate(merged)
