"""Planner-first 路由协议检查，复用 core/routing_eval.ROUTING_CASES。

不依赖 LLM / ChromaDB / 网络。所有自然语言样例都必须声明为先经过任务规划器。

Run:
  python tests/test_routing.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator_agent import OrchestratorAgent
from core.routing_eval import ROUTING_CASES, decide_case


def main() -> int:
    orch = object.__new__(OrchestratorAgent)
    total = len(ROUTING_CASES)
    failed = 0
    current_cat = None
    for case in ROUTING_CASES:
        if case["category"] != current_cat:
            current_cat = case["category"]
            print(f"\n== {current_cat} ==")
        passed, detail = decide_case(orch, case)
        if not passed:
            failed += 1
        print(f"[{'PASS' if passed else 'FAIL'}] {case['query']}: {detail}")
    print(f"\n共 {total} 例, 通过 {total - failed} 例, 失败 {failed} 例")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
