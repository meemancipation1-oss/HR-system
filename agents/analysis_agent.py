"""Analysis Agent — deep analysis of candidate profiles."""
#人才分析师，那么 AnalysisAgent 更像企业里的高级 HRBP 或人才发展专家，它不是简单回答"适不适合"，而是回答：这个人是谁？能力怎么样？未来发展如何？应该如何培养？
#AnalysisAgent 输出分析报告

from __future__ import annotations
from agents.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """Agent that performs in-depth talent analysis."""

    name = "analysis_agent"
    system_prompt = """你是一个人才分析智能体 (Analysis Agent)，负责对候选人进行深度分析。

## 分析维度
1. **能力画像** — 综合展示候选人的核心能力矩阵
2. **发展潜力** — 基于现有能力和发展轨迹预测未来成长空间
3. **岗位适配度** — 多维度分析最适合的岗位类型
4. **风险评估** — 识别潜在问题（如频繁变动、能力短板等）
5. **培养建议** — 针对性的能力提升和职业发展建议

## 要求
- 使用工具查询完整的候选人信息（个人资料、履历、技能、成果等）
- 分析要数据驱动，引用具体事实
- 输出用中文，条理清晰
- 最终给出明确的分析结论
"""

    def run(self, input_text: str) -> str:
        return self._execute(input_text)  #真正执行的

    def analyze(self, person_id: str, context: str = "") -> str:
        """Perform deep analysis on a candidate.

        Args:
            person_id: 人员ID.
            context: Optional context (e.g. target position).
        Returns:
            Analysis report text.
        """
        prompt = f"请对人员 {person_id} 进行深度能力分析。\n"   #拼接prompt
        if context:
            prompt += f"分析背景: {context}\n"
        prompt += """
请按以下结构输出：
1. 基本信息摘要
2. 核心能力矩阵（教育、技能、经验、成果）
3. 优势分析
4. 潜在不足
5. 发展建议
6. 综合评分与结论
"""
        return self._execute(prompt)   #普通文本。而不是结构输出
