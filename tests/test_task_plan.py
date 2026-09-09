"""Planner-first orchestration tests without LLM, vector DB, or CSV access."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator_agent import OrchestratorAgent
from core.memory import AgentMemory
from models.task_plan import HiringTaskPlan


class PlannerFirstTests(unittest.TestCase):
    def test_plan_bounds_candidate_count(self) -> None:
        with self.assertRaises(ValidationError):
            HiringTaskPlan(actions=["search"], candidate_count=0)

    def test_every_chat_request_creates_a_plan_before_execution(self) -> None:
        orch = object.__new__(OrchestratorAgent)
        orch._memory = AgentMemory()
        observed = []

        def create_plan(text: str) -> HiringTaskPlan:
            observed.append(("plan", text))
            return HiringTaskPlan(actions=["data_query"])

        def execute(plan: HiringTaskPlan, text: str) -> str:
            observed.append(("execute", plan.actions, text))
            return "已执行"

        orch._create_task_plan = create_plan
        orch._execute_task_plan = execute

        for query in ("找 3 名 Python 后端候选人", "不要本科生"):
            result, metrics, agent = orch.run_with_metrics(query)
            self.assertEqual(result, "已执行")
            self.assertEqual(agent, "task_planner")
            self.assertEqual(metrics["planner_llm_calls"], 1)

        self.assertEqual(
            observed,
            [
                ("plan", "找 3 名 Python 后端候选人"),
                ("execute", ["data_query"], "找 3 名 Python 后端候选人"),
                ("plan", "不要本科生"),
                ("execute", ["data_query"], "不要本科生"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
