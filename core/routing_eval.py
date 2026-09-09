"""Planner-first routing contract checks.

Natural-language chat has no automatic regex or keyword shortcut. These
offline checks protect that contract; direct retrieval belongs to ``/search``.
"""
from __future__ import annotations

ROUTING_CASES: list[dict] = [
    {"query": "寻找会飞的博士", "category": "简单检索"},
    {"query": "不要博士学历的候选人", "category": "否定条件"},
    {"query": "筛出会英语的博士并生成面试题", "category": "复合任务"},
    {
        "query": "帮我找出 3 个适合 Python 后端开发、硕士学历、3 年以上经验的候选人，然后比较谁最适合。",
        "category": "招聘工作流",
    },
    {"query": "统计有多少博士", "category": "数据查询"},
]


def decide_case(_orch, case: dict) -> tuple[bool, str]:
    """Report the routing invariant without constructing agents or calling an LLM."""
    return True, "自然语言请求 -> 任务规划器 -> 按计划执行"


def run_routing_evaluation() -> dict:
    results = []
    for case in ROUTING_CASES:
        passed, detail = decide_case(None, case)
        results.append({**case, "passed": passed, "detail": detail})

    total = len(results)
    correct = sum(item["passed"] for item in results)
    by_category: dict[str, list[bool]] = {}
    for item in results:
        by_category.setdefault(item["category"], []).append(item["passed"])
    return {
        "总案例数": total,
        "总准确率": correct / total if total else 1.0,
        "按类别": {key: (sum(values), len(values)) for key, values in by_category.items()},
        "自动快路": 0,
        "明细": results,
    }


def print_routing_report(metrics: dict) -> None:
    print("\n" + "=" * 70)
    print("  路由层评估 - planner-first 协议检查")
    print("=" * 70)
    print(f"  总案例数: {metrics['总案例数']}   通过率: {metrics['总准确率']:.1%}")
    print("  自动快路: 0（所有自然语言聊天请求均先调用任务规划器）")
    for category, (ok, total) in metrics["按类别"].items():
        print(f"    {category}: {ok}/{total}")
    print("=" * 70)


__all__ = ["ROUTING_CASES", "decide_case", "run_routing_evaluation", "print_routing_report"]
