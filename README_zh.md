<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/langchain-workflow-blueviolet" alt="LangChain">
  <img src="https://img.shields.io/badge/chroma-vectorstore-green" alt="Chroma">
  <img src="https://img.shields.io/badge/mysql-adapter-orange" alt="MySQL">
  <img src="https://img.shields.io/badge/license-MIT-success" alt="License">
</p>

## 概览
面向钢包预热等工业过程的 LLM 决策支持系统。包含元认知 Orchestrator、基于反馈的检索链路、记忆增强 Text-to-SQL、本地轻量工具与 CLI 入口。

## 主要特性
- 元认知路由：深度/分支/工具链预算控制，路由到检索 / SQL / 工具；工具效用融合 LLM 预测与历史成功率。
- 检索：层次化切分、Q2Q 合成查询、Chroma + BM25 混合检索、自适应 Top-K、Qwen rerank、反馈增量更新。
- SQL 记忆：Chroma 记忆库存放 DDL/文档/参考 SQL 与交互记录；DeepSeek 生成 + MySQL 执行，最多 6 轮修正。
- CLI：`python -m src.indu_cognition.cli.run_agent --query "..."`；配置通过 `configs/default.yaml` 和 `configs/providers.yaml`/`.env`。

## 快速开始
```bash
conda activate inducognition-llm
python -m src.indu_cognition.cli.run_agent --query "查看昨日包龄统计"
```
将 API Key 写入 `.env` 或 `configs/providers.yaml`。

## 目录
- `src/indu_cognition/config`：Pydantic 配置与加载
- `src/indu_cognition/llm`：DeepSeek/Qwen 适配，G-Eval 包装
- `src/indu_cognition/retrieval`：切分、Q2Q、Chroma、BM25、混合检索、反馈更新
- `src/indu_cognition/sql_memory`：记忆库、prompt 构造、Text-to-SQL、MySQL 执行
- `src/indu_cognition/agent`：路由、工具注册、Orchestrator 主循环
- `src/indu_cognition/tools`：本地轻量工具
- `configs/`：默认配置与 provider 示例
- `storage/`：向量库持久化（已忽略）
