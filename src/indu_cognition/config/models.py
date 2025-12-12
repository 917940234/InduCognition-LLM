"""配置模型定义。

覆盖设计文档的超参数与运行参数，将默认值集中管理，便于复现与调优。
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM 调用相关配置。"""

    model_config = {"extra": "ignore"}

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.8
    top_k: int = 40
    max_tokens: int = 4096
    frequency_penalty: float = 1.1


class EmbeddingConfig(BaseModel):
    """向量模型配置。"""

    model_config = {"extra": "ignore"}

    model: str = "text-embedding-v3"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: Optional[str] = None


class RerankConfig(BaseModel):
    """Rerank 模型配置。"""

    model_config = {"extra": "ignore"}

    model: str = "qwen3-rerank"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: Optional[str] = None
    top_n: int = 50


class AgentConfig(BaseModel):
    """元认知 Orchestrator 控制超参。"""

    model_config = {"extra": "ignore"}

    d_max: int = 3
    b_max: int = 4
    routing_threshold: float = Field(0.65, description="η_route")
    tool_utility_threshold: float = Field(0.55, description="δ_tool")
    max_tool_chain: int = Field(3, description="L_tool")
    tool_timeout_sec: int = Field(20, description="t_max")
    max_iterations: int = Field(6, description="Agent Max Iterations")
    alpha_tool: float = Field(0.6, description="工具评分中 LLM 预测权重")


class RetrievalConfig(BaseModel):
    """检索与分段超参。"""

    model_config = {"extra": "ignore"}

    parent_delimiter: str = "\n\n"
    child_delimiters: List[str] = Field(default_factory=lambda: ["\n", ",", "."])
    parent_max_tokens: int = 500
    child_max_tokens: int = 200
    overlap: int = 50
    top_k: int = 3
    score_threshold: float = 0.5
    lambda_hybrid: float = 0.5
    gamma_filter: float = 1.0
    expected_chunk_len: int = 200
    llm_context_len: int = 4096


class VectorStoreConfig(BaseModel):
    """向量库配置。"""

    model_config = {"extra": "ignore"}

    persist_path: str = "storage/chroma_docs"
    collection: str = "docs"
    embedding_model: str = "text-embedding-v3"


class BM25Config(BaseModel):
    """BM25 配置占位。"""

    model_config = {"extra": "ignore"}

    k1: float = 1.5
    b: float = 0.75


class SQLMemoryConfig(BaseModel):
    """SQL 记忆库向量存储配置。"""

    model_config = {"extra": "ignore"}

    persist_path: str = "storage/chroma_sqlmem"
    collection: str = "sqlmem"


class SQLConfig(BaseModel):
    """Text-to-SQL 相关配置。"""

    model_config = {"extra": "ignore"}

    t_max: int = 6
    execution_timeout_sec: int = 20
    max_rows: int = 1000
    dsn: Optional[str] = None
    memory_store: SQLMemoryConfig = SQLMemoryConfig()


class EvaluationConfig(BaseModel):
    """评估配置。"""

    model_config = {"extra": "ignore"}

    bertscore_model: str = "microsoft/deberta-base-mnli"
    use_g_eval: bool = True
    g_eval_model: str = "deepseek-chat"
    applause_rubric: dict = Field(
        default_factory=lambda: {
            "accuracy": 0.4,
            "relevance": 0.3,
            "completeness": 0.15,
            "clarity": 0.1,
            "language_grammar": 0.05,
        }
    )
    g_eval_api_key: Optional[str] = None


class LoggingConfig(BaseModel):
    """日志配置。"""

    model_config = {"extra": "ignore"}

    level: str = "INFO"
    json: bool = False
    log_dir: Optional[str] = None


class AppConfig(BaseModel):
    """全局应用配置。"""

    model_config = {"extra": "ignore"}

    agent: AgentConfig = AgentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    rerank: RerankConfig = RerankConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    bm25: BM25Config = BM25Config()
    sql: SQLConfig = SQLConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig = LoggingConfig()
