"""Regression tests for the compound hiring orchestration path.

No LLM, ChromaDB, or CSV files are needed: domain agents are replaced by
small fakes so the test verifies the IDs and evidence crossing stage borders.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator_agent import OrchestratorAgent
from agents.screening_agent import ScreeningOutput
from core.memory import AgentMemory
from models.task_plan import HiringTaskPlan


class FakeDataAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search_semantic(self, query: str, k: int) -> list[dict]:
        self.calls.append((query, k))
        return [
            {"人员ID": "1", "姓名": "甲", "摘要": "候选人甲"},
            {"人员ID": "2", "姓名": "乙", "摘要": "候选人乙"},
            {"人员ID": "3", "姓名": "丙", "摘要": "候选人丙"},
            {"人员ID": "4", "姓名": "丁", "摘要": "候选人丁"},
        ]


class FakeScreeningAgent:
    outcomes = {
        "1": (80, "pass"),
        "2": (92, "review"),
        "3": (99, "reject"),
        "4": (88, "pass"),
    }

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def screen_candidate(self, person_id: str, requirements: str = "") -> ScreeningOutput:
        self.calls.append((person_id, requirements))
        score, decision = self.outcomes[person_id]
        return ScreeningOutput(
            评估思路="测试",
            匹配分数=score,
            技能评分=score,
            经验评分=score,
            教育评分=score,
            综合评价="测试",
            匹配职位="Python 后端开发",
            匹配理由="测试证据",
            决策=decision,
            证据=[f"候选人 {person_id} 的档案证据"],
            需要人工复核=decision == "review",
        )


class FakeAnalysisAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def analyze(self, person_id: str, context: str = "") -> str:
        self.calls.append((person_id, context))
        return f"候选人 {person_id} 的深度分析"


class FakeReportAgent:
    def __init__(self) -> None:
        self.memory = AgentMemory()
        self.prompts: list[str] = []

    def run(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "最终比较报告"


class HiringPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orch = object.__new__(OrchestratorAgent)
        self.orch._memory = AgentMemory()
        self.data = FakeDataAgent()
        self.screening = FakeScreeningAgent()
        self.analysis = FakeAnalysisAgent()
        self.report = FakeReportAgent()
        self.orch._agents = {
            "data_agent": self.data,
            "screening_agent": self.screening,
            "analysis_agent": self.analysis,
            "report_agent": self.report,
        }

    def test_compound_request_uses_staged_pipeline_and_excludes_rejections(self) -> None:
        query = (
            "帮我找出 3 个适合 Python 后端开发、硕士学历、3 年以上经验的候选人，"
            "然后比较一下这三个人谁最适合我们这个岗位。"
        )

        planner_calls = []
        self.orch._create_task_plan = lambda text: (
            planner_calls.append(text) or HiringTaskPlan(
                actions=["search", "screen", "analyze", "compare"],
                candidate_count=3,
                position_name="Python 后端开发",
                requirements=["硕士学历", "3 年以上经验"],
            )
        )

        result = self.orch.run(query)

        self.assertEqual(planner_calls, [query])
        self.assertEqual(self.data.calls[0][1], 12)
        self.assertEqual([call[0] for call in self.screening.calls], ["1", "2", "3", "4"])
        # Candidate 3 has the highest numeric score but is rejected, so it is
        # excluded before analysis and report ranking.
        self.assertEqual([call[0] for call in self.analysis.calls], ["2", "4", "1"])
        self.assertIn("候选人ID：2, 4, 1", self.report.prompts[0])
        self.assertNotIn('"人员ID": "3"', self.report.prompts[0])
        self.assertIn("已从 4 名召回候选人中完成筛选，选出 3 名进入比较。", result)
        self.assertIn("最终比较报告", result)
        self.assertEqual(len(self.orch.get_memory_snapshot()), 2)

    def test_exclusion_is_forwarded_to_screening(self) -> None:
        plan = HiringTaskPlan(
            actions=["search", "screen", "analyze", "compare"],
            candidate_count=3,
            requirements=["Python 后端开发", "3 年以上经验"],
            exclusions=["学历为本科"],
        )

        self.orch._run_hiring_pipeline("找合适的后端候选人，不要本科生", plan)

        requirements = self.screening.calls[0][1]
        self.assertIn("Python 后端开发", requirements)
        self.assertIn("排除条件：学历为本科", requirements)


if __name__ == "__main__":
    unittest.main()
