<p align="center">
  <strong>English</strong> | <a href="README_zh.md">中文</a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/langchain-workflow-blueviolet" alt="LangChain">
  <img src="https://img.shields.io/badge/chroma-vectorstore-green" alt="Chroma">
  <img src="https://img.shields.io/badge/mysql-adapter-orange" alt="MySQL">
  <img src="https://img.shields.io/badge/license-MIT-success" alt="License">
</p>

## Overview
Industrial LLM decision-support system for ladle preheating and related processes. It bundles a meta-cognitive orchestrator, feedback-driven retrieval, memory-augmented Text-to-SQL, lightweight tools, and CLI entrypoints.

## Highlights
- Meta routing with depth/branch/tool budgets; routes to retrieval / SQL / tools. Tool utility blends LLM scores and historical success.
- Retrieval: hierarchical chunking, Q2Q synthetic queries, Chroma + BM25 hybrid search, adaptive Top-K, Qwen rerank, feedback-aware incremental updates.
- SQL memory: Chroma-based store for DDL/docs/reference SQL & interaction logs; DeepSeek generation + MySQL execution with up to 6 repair rounds.
- CLI: `python -m src.indu_cognition.cli.run_agent --query "..."`; config via `configs/default.yaml` plus `configs/providers.yaml`/`.env`.

## Quickstart
```bash
conda activate inducognition-llm
python -m src.indu_cognition.cli.run_agent --query "show ladle ageing stats"
```
Place API keys in `.env` or `configs/providers.yaml`.

## Layout
- `src/indu_cognition/config`: Pydantic configs & loader
- `src/indu_cognition/llm`: DeepSeek/Qwen adapters, G-Eval wrapper
- `src/indu_cognition/retrieval`: chunking, Q2Q, Chroma, BM25, hybrid retrieval, feedback updates
- `src/indu_cognition/sql_memory`: memory store, prompt builder, Text-to-SQL, MySQL executor
- `src/indu_cognition/agent`: routing, tool registry, orchestrator loop
- `src/indu_cognition/tools`: lightweight local tools
- `configs/`: defaults and provider examples
- `storage/`: vector store persistence (git-ignored)
