#招聘经理，汇总所有分析结果，形成最终的招聘报告
"""Report Agent — generates comprehensive recruitment reports."""
from __future__ import annotations
from datetime import datetime

from agents.base_agent import BaseAgent
from models.candidate import RecruitmentReport


class ReportAgent(BaseAgent):
    """Agent that generates structured recruitment reports."""

    name = "report_agent"
    system_prompt = """你是一个报告生成智能体 (Report Agent)，负责生成专业的招聘分析报告。

## 报告类型
1. **单候选人评估报告** — 深度分析单个候选人的优劣势
2. **岗位匹配报告** — 多候选人与岗位的匹配对比
3. **招聘总结报告** — 整个招聘流程的汇总分析
4. **数据洞察报告** — 基于HR数据的人才趋势分析

## 要求
- 使用工具获取数据支持
- 报告结构清晰，专业规范
- 包含数据可视化的描述建议
- 给出可操作的决策建议
- 用中文输出
"""

    def run(self, input_text: str) -> str:  #调用户
        return self._execute(input_text)

    def generate_recruitment_report(
        self,
        position_name: str,
        candidate_ids: list[str],
        additional_context: str = "",
    ) -> RecruitmentReport:
        """Generate a full recruitment report.

        Args:
            position_name: The target position name.
            candidate_ids: List of candidate 人员IDs to evaluate.
            additional_context: Extra context for the report.
        Returns:
            Structured RecruitmentReport.
        """
        prompt = f"""请为以下招聘需求生成完整的招聘报告。

目标岗位: {position_name}
候选人列表: {', '.join(candidate_ids)}
"""
        if additional_context:
            prompt += f"附加信息: {additional_context}\n"

        prompt += """
请逐步执行：
1. 查询每个候选人的详细信息、技能、履历
2. 查询相关岗位信息
3. 对每个候选人进行评分和排名
4. 给出最终推荐意见和招聘建议

请用中文输出完整报告。
"""
        output = self._execute(prompt)

        return RecruitmentReport(
            职位名称=position_name,
            候选人总数=len(candidate_ids),
            推荐候选人=[],
            ai_分析总结=output,
        )
