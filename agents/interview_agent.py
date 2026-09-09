#面试官，根据候选人背景生成有针对性的面试问题，根据候选人的回答进行评分和评价
"""Interview Agent — generates interview questions and evaluates responses."""
from __future__ import annotations
import re
import json

from agents.base_agent import BaseAgent
from models.candidate import InterviewQuestion


class InterviewAgent(BaseAgent):
    """Agent that conducts AI-powered interviews."""

    name = "interview_agent"
    system_prompt = """你是一个面试智能体 (Interview Agent)，负责生成面试问题和评估回答。

## 你的能力
1. **问题生成** — 根据候选人背景和岗位要求生成有针对性的面试问题
2. **回答评估** — 评估候选人对问题的回答质量
3. **面试报告** — 生成面试总结报告

## 问题类型
- 技术能力题 — 考察专业技能和项目经验
- 情景题 — 考察应变能力和决策能力
- 行为题 — 考察团队协作和领导力
- 综合素质题 — 考察沟通、逻辑、价值观

## 要求
- 使用工具查询候选人背景信息
- 问题要因人而异，有针对性
- 评估要客观公正
- 用中文交流
"""

    def run(self, input_text: str) -> str:
        return self._execute(input_text)

#最重要的方法
    def generate_questions(
        self,
        person_id: str,
        position_info: str = "",  #岗位描述
        count: int = 5,  #问题个数
    ) -> list[InterviewQuestion]:
        """Generate tailored interview questions.

        Args:
            person_id: 人员ID.
            position_info: Optional position description.
            count: Number of questions to generate.
        Returns:
            List of InterviewQuestion objects.
        """
        prompt = f"请为候选人 {person_id} 生成 {count} 个面试问题。\n"
        if position_info:
            prompt += f"目标岗位信息: {position_info}\n"

        prompt += """
请先查询候选人的详细背景信息，然后根据其背景生成有针对性的问题。

对每个问题，请给出：
- 问题内容
- 考察点（考察什么能力）
- 难度等级（简单/中等/困难）

请按以下JSON格式输出：
```json
[
  {"问题": "...", "考察点": "...", "难度": "中等"},
  ...
]
```
"""
        output = self._execute(prompt)  #调用LLM
        # Try to parse structured output，用re提取"""中的Json"""（大模型很爱输出带引号的）
        try:
            match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
            if match:
                data = json.loads(match.group(1))  #提取成python列表
            else:
                data = json.loads(output)
            return [InterviewQuestion(**q) for q in data]
        except Exception:
            return [InterviewQuestion(问题=output, 考察点="综合能力")]

    def evaluate_answer(self, question: str, answer: str, context: str = "") -> str:
        """Evaluate a candidate's answer.

        Args:
            question: The question asked.
            answer: The candidate's answer.
            context: Optional context (person profile).
        Returns:
            Evaluation text.
        """
        prompt = f"""请评估以下面试回答。

面试问题: {question}
候选人的回答: {answer}
"""
        if context:
            prompt += f"候选人背景: {context}\n"

        prompt += """
请从以下维度评估：
1. 回答的完整性和准确性
2. 逻辑性和条理性
3. 专业深度
4. 与岗位的匹配度
5. 总体评分 (A/B/C/D)
6. 改进建议
"""
        return self._execute(prompt)  #自然语言输出给人看
