"""命令行入口：单条查询运行 Orchestrator。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..agent import build_orchestrator
from ..cli.logging.setup import setup_logging
from ..config import load_app_config
from ..llm import build_llm_clients


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InduCognition Orchestrator for a single query.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--providers", type=Path, default=Path("configs/providers.yaml"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--query", type=str, required=True, help="User query to process.")
    args = parser.parse_args()

    setup_logging()
    app_config = load_app_config(args.config, providers_path=args.providers, env_path=args.env)
    llm_clients = build_llm_clients(app_config)
    orchestrator = build_orchestrator(app_config, llm_clients)
    state = orchestrator.run(args.query)
    print("== Response ==")
    print(state.response)


if __name__ == "__main__":
    main()
