"""Data Agent — ingests, indexes, and explores the HR dataset."""
#Data Agent：负责导入、建立索引、探索 HR 数据集。
#数据库管理员
from __future__ import annotations  #延迟解析类型注解
from agents.base_agent import BaseAgent
from core.rag import RAGEngine


class DataAgent(BaseAgent):
    """Agent responsible for data ingestion and exploration."""

    name = "data_agent"
    system_prompt = """你是一个 HR 数据智能体 (Data Agent)，负责管理和探索人力资源数据库。

你可以访问以下工具：
- search_individual: 搜索人员信息
- get_individual_detail: 获取人员详细信息
- search_candidates_all_tables: 跨所有人员关联表召回候选人（语言、任职、专利等）
- search_table: 按关键词查询任意 HR CSV 表
- list_available_tables: 列出所有数据表
- 以及其他检索工具

你的职责包括：
1. 回答关于数据内容和结构的问题
2. 搜索特定人员或职位的信息
3. 提供数据统计概览
4. 解释各数据表之间的关系
请用中文回答，保持专业且准确。"""

    def __init__(self) -> None:
        super().__init__()
        self.rag = RAGEngine()   #RAGEngine实现数据导入（ingest）和建索引（index）

    def run(self, input_text: str) -> str: #Agent 的入口
        return self._execute(input_text)   #子类只提供 Prompt 和工具，执行流程统一由父类实现。

    def explore_dataset(self) -> str:   #读取配置。
        """Provide a structured overview of the entire dataset."""
        from config.settings import AVAILABLE_TABLES
        lines = ["## HR 数据集概览\n"]
        for tbl, desc in AVAILABLE_TABLES.items():
            lines.append(f"- **{tbl}**: {desc}")
        return "\n".join(lines)

#这是整个 DataAgent 最重要的方法。真正干活的是RAGEngine.search()
    def search_semantic(
        self,
        query: str,
        k: int = 5,
        top_k: int | None = None,
    ) -> list[dict]:
        """Semantic search over candidates using vector embeddings."""
        # ``top_k`` is retained for callers using the original API; ``k`` is
        # the spelling used by the Orchestrator and direct search endpoint.
        return self.rag.search(query, k=top_k if top_k is not None else k)

    def search_candidates_all_tables(self, query: str, k: int = 20) -> list[dict]:
        """Recall candidate IDs from every CSV containing ``人员ID``."""
        return self.rag.search_candidates_in_all_tables(query, k=k)

    def should_use_structured_search(self, query: str) -> bool:
        """Return whether the query contains an associated-table condition."""
        return self.rag.has_structured_hint(query)

#因此，DataAgent 本身的代码其实非常少，它主要承担的是配置和调度的职责：
#system_prompt：告诉大模型它是谁、能做什么。
#run()：把请求交给 BaseAgent 的统一执行流程。
#explore_dataset()：返回数据集结构概览，不需要 LLM。
#search_semantic()：把语义搜索请求转发给 RAGEngine，由 RAG 完成向量检索并返回结果。
#真正复杂的逻辑（如 LLM 对话循环、Tool Calling、RAG 检索）都分别封装在 BaseAgent 和 RAGEngine 中，DataAgent 更像是连接这些组件的桥梁。"""
