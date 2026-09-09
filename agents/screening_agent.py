"""Screening Agent — matches candidates to positions with structured scoring.

Uses forced tool calling for guaranteed structured output (since DeepSeek
doesn't support response_format=json_object natively).
#简历筛选专家，根据候选人的信息和岗位要求，对候选人进行评分、匹配，并输出结构化的评估结果。输出固定评分
输出的是固定结构（Structured Output），而不是普通文本。
会先调用工具查询候选人与岗位信息，再进行评分。
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent

#Field 的作用：① 给 LLM 提示 ② 给程序做验证
#继承BaseModel，ScreeningAgent：只负责：告诉 BaseAgent：我是干什么的。输出什么。
class ScreeningOutput(BaseModel):
    """Structured output for screening results.继承自BaseModel:规定 LLM 最终必须返回什么格式。"""
    评估思路: str = Field(description="评估思路和考量因素")
    匹配分数: float = Field(description="综合匹配分数 (0-100)", ge=0, le=100)
    技能评分: float = Field(description="技能匹配度评分", ge=0, le=100)
    经验评分: float = Field(description="经验匹配度评分", ge=0, le=100)
    教育评分: float = Field(description="教育背景评分", ge=0, le=100)
    综合评价: str = Field(description="综合评价意见")
    匹配职位: str = Field(description="推荐匹配的职位名称")
    匹配理由: str = Field(description="为什么匹配/不匹配")
    数据矛盾: str = Field(default="", description="若档案字段之间互相矛盾(如学历与学位字段不一致), 注明冲突字段与采信依据; 无矛盾则留空")
    决策: Literal["pass", "reject", "review"] = Field(
        default="review", description="pass=推荐, reject=不推荐, review=需要人工复核"
    )
    证据: list[str] = Field(default_factory=list, description="支持结论的候选人事实或工具来源")
    缺失条件: list[str] = Field(default_factory=list, description="岗位要求中未满足或无法确认的条件")
    需要人工复核: bool = Field(default=True, description="字段冲突、证据不足或边界分数时必须为 true")


class ScreeningAgent(BaseAgent):
    """Agent that screens and scores candidates against positions."""

    name = "screening_agent"
    system_prompt = """你是一个人才筛选智能体 (Screening Agent)，负责将候选人与职位进行精确匹配。

## 你的评估维度
1. **教育背景** (学历、院校、专业与职位要求的匹配度)
2. **专业技能** (职业资格、技术能力、擅长领域)
3. **工作经验** (任职履历、项目经历、管理经验)
4. **综合能力** (沟通协调、逻辑分析、抽象思维、人格特质)
5. **特殊条件** (专利、论文、奖项、语言能力等加分项)

## 评估原则
- 使用工具查询候选人完整信息
- 使用工具查询职位的必备/禁止/可选条件
- 综合所有维度给出 0-100 的评分
- 用中文回答

## 评分纪律（必须遵守）
- 所有结论必须来自工具返回的字段，不得推测、补全或美化档案中不存在的信息
- `学历`/`受教育程度` 是候选人学历的唯一权威字段；`本科学位/硕士学位/博士学位`、`本科毕业院校/硕士毕业院校/博士毕业院校` 等字段若与权威字段矛盾，不得据此认定候选人拥有更高学历
- 若发现档案字段互相矛盾，必须在输出 `数据矛盾` 字段中说明冲突字段与采信依据，评分以权威字段为准
- 逐项对照职位要求的硬性条件给出评分，不要凭整体印象随意加减分"""

    def run(self, input_text: str) -> str:
        return self._execute_structured(input_text, ScreeningOutput)#要求 LLM 按照 ScreeningOutput 这个 Pydantic 模型生成结果。

    def screen_candidate(  #程序接口（Programmatic API）
        self,
        person_id: str,
        position_id: str | None = None,
        requirements: str = "",
    ) -> ScreeningOutput:
        """Screen a specific candidate against requirements (programmatic API).

        Returns a structured ScreeningOutput object.
        """
        prompt = f"请评估以下候选人是否适合给定的职位要求。\n候选人ID: {person_id}\n"
        if position_id:
            prompt += f"职位ID: {position_id}\n"
        if requirements:
            prompt += f"职位要求: {requirements}\n"
        prompt += "\n请先使用工具查询完整信息，然后给出评估结果。"

        return self._execute_structured(prompt, ScreeningOutput)
