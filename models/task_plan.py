"""Validated task plans produced before executing a natural-language request."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TaskAction = Literal[
    "data_query",
    "search",
    "screen",
    "analyze",
    "interview",
    "compare",
    "report",
]


class HiringTaskPlan(BaseModel):
    """Execution contract between the planner and the orchestration layer."""

    actions: list[TaskAction] = Field(
        default_factory=lambda: ["data_query"],
        description="Actions to execute in dependency order.",
    )
    candidate_count: int = Field(default=5, ge=1, le=10)
    candidate_ids: list[str] = Field(default_factory=list)
    position_name: str = ""
    requirements: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    clarification_needed: bool = False
    clarification_question: str = ""


PLANNER_SYSTEM_PROMPT = """你是招聘任务规划器。将用户请求转为 HiringTaskPlan，不能执行招聘判断。

可用 actions：
- data_query：查询数据集、统计或解释数据
- search：召回候选人
- screen：逐人核验岗位条件并给出带证据的结构化筛选结论
- analyze：对入围者做能力、风险和潜力分析
- interview：为入围者生成面试题
- compare：比较多个候选人并排序
- report：生成招聘报告

规则：
1. 所有自然语言中的硬性条件都放入 requirements；排除/否定条件必须放入 exclusions。
2. 对“不要本科生”写入 exclusions，例如“学历为本科”；不得遗漏否定条件。
3. 找人后再比较/推荐时，actions 至少为 [search, screen, analyze, compare]。
   search 会覆盖候选人主档案及任职、语言、成果、军事经历等关联表。
   条件只存在于关联表时也必须先使用 search，不要直接假设主档案中存在该字段。
4. 候选人数量未说明时，找人默认 5；比较时默认 3。
5. 已给出的人员ID写入 candidate_ids。信息不足且无法合理执行时，设置 clarification_needed=true 并提供问题。
6. 只调用 output_result 输出符合 schema 的结果。"""


__all__ = ["HiringTaskPlan", "PLANNER_SYSTEM_PROMPT", "TaskAction"]
