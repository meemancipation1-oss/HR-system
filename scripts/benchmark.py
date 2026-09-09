"""Benchmark script — evaluates system performance and generates metrics.

Run:  python scripts/benchmark.py

Measures:
  - Standard Agent loop path (with keyword classification)
  - Semantic search shortcut (bypasses Agent loop)
"""

from __future__ import annotations
import sys
import time
import statistics
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator_agent import OrchestratorAgent


BENCHMARKS = [
    # Shortcut path uses a search intent query
    {"name": "语义搜索(短路-RAG直查)", "input": "寻找有博士学历且有发明专利的候选人", "mode": "shortcut"},
    # Regular path forces Agent loop via run_pipeline (bypasses shortcut detection)
    {"name": "语义搜索(常规-Agent循环)", "input": "有博士学历且有发明专利的候选人", "mode": "agent_loop"},
    {"name": "候选人筛选", "input": "请筛选评估候选人 1 是否适合高级技术研发岗位。要求：博士学历", "mode": "normal"},
    {"name": "深度分析", "input": "请对候选人 1 进行深度能力分析", "mode": "normal"},
    {"name": "面试生成", "input": "请为候选人 1 生成 5 个面试问题", "mode": "normal"},
    {"name": "招聘报告", "input": "请为岗位【技术总监】生成招聘报告，候选人为人员ID=1,2,3", "mode": "normal"},
]


def run_benchmark(orch: OrchestratorAgent, rounds: int = 3) -> list[dict]:
    results = []

    for case in BENCHMARKS:
        print(f"\n{'='*55}")
        print(f"📋 {case['name']}")
        print(f"   Input: {case['input'][:60]}...")

        round_times = []
        round_llm = []
        round_tools = []
        round_agents = []

        for r in range(rounds):
            print(f"   Round {r+1}/{rounds}...", end=" ")
            try:
                if case["mode"] == "shortcut":
                    # Direct RAG — bypass Agent entirely
                    t0 = time.time()
                    orch.search_semantic(case["input"], k=5)
                    elapsed = (time.time() - t0) * 1000
                    round_times.append(elapsed)
                    round_llm.append(0)
                    round_tools.append(0)
                    round_agents.append("data_agent(RAG)")
                    print(f"✓ {elapsed:.0f}ms | RAG直查 (0 LLM)")

                elif case["mode"] == "agent_loop":
                    # Force agent loop by running through run_pipeline
                    # using data_agent which has a normal tool loop
                    t0 = time.time()
                    result = orch.run_pipeline("data_agent", f"请搜索{case['input']}")
                    elapsed = (time.time() - t0) * 1000
                    round_times.append(elapsed)
                    # We can't track metrics through run_pipeline, so estimate
                    round_llm.append("?")
                    round_tools.append("?")
                    round_agents.append("data_agent")
                    print(f"✓ {elapsed:.0f}ms | Agent循环")

                else:
                    _, metrics, agent = orch.run_with_metrics(case["input"])
                    round_times.append(metrics["elapsed_ms"])
                    round_llm.append(metrics.get("llm_calls", 0))
                    round_tools.append(metrics.get("tool_calls", 0))
                    round_agents.append(agent)
                    print(f"✓ {metrics['elapsed_ms']:.0f}ms | LLM:{metrics.get('llm_calls',0)} 工具:{metrics.get('tool_calls',0)}")

            except Exception as e:
                print(f"✗ {e}")

        if round_times:
            results.append({
                "name": case["name"],
                "avg_ms": round(statistics.mean(round_times), 1),
                "min_ms": round(min(round_times), 1),
                "max_ms": round(max(round_times), 1),
                "avg_llm": round(statistics.mean([x for x in round_llm if isinstance(x, (int, float))]), 1) if any(isinstance(x, (int, float)) for x in round_llm) else "?",
                "avg_tools": round(statistics.mean([x for x in round_tools if isinstance(x, (int, float))]), 1) if any(isinstance(x, (int, float)) for x in round_tools) else "?",
                "agent": round_agents[0] if round_agents else "—",
            })

    return results


def print_results(results: list[dict]):
    print(f"\n\n{'='*75}")
    print(f"  📊 HR Multi-Agent System — Performance Benchmark")
    print(f"{'='*75}")
    print(f"  {'Scenario':<35} {'Avg':>8} {'Min':>8} {'Max':>8} {'LLM':>5} {'Tools':>5}")
    print(f"  {'─'*71}")

    for r in results:
        llm_str = f"{r['avg_llm']}x" if isinstance(r['avg_llm'], (int, float)) else "?"
        tools_str = f"{r['avg_tools']}x" if isinstance(r['avg_tools'], (int, float)) else "?"
        print(f"  {r['name']:<35} {r['avg_ms']:>6.0f}ms {r['min_ms']:>6.0f}ms {r['max_ms']:>6.0f}ms {llm_str:>4} {tools_str:>4}")

    print(f"  {'─'*71}")

    # Find shortcut vs regular
    shortcut = next((r for r in results if "短路" in r["name"]), None)
    regular = next((r for r in results if "常规" in r["name"]), None)

    if shortcut and regular:
        improvement = (regular["avg_ms"] - shortcut["avg_ms"]) / regular["avg_ms"] * 100
        print(f"\n  🏆 搜索短路收益：{improvement:.0f}% 延迟下降 (Agent循环 → RAG直查)")

    # Average of normal scenarios (excluding shortcut/regular comparison)
    normal = [r for r in results if "短路" not in r["name"] and "常规" not in r["name"]]
    if normal:
        avg = statistics.mean([r["avg_ms"] for r in normal])
        print(f"  非搜索场景平均延迟: {avg:.0f}ms")

    print(f"\n  🖥  Model: DeepSeek Chat")
    print(f"  🔧  Agent分类: 关键词匹配(0 LLM) + LLM兜底")
    print(f"  ⚡  搜索短路: 检测搜索意图→直查ChromaDB")
    print(f"  🔄  并行工具: ThreadPoolExecutor多工具并行执行")
    print(f"{'='*75}")


def main():
    print("=" * 75)
    print("  HR Multi-Agent System — Performance Benchmark")
    print("  Running each scenario 3 times for stable averages...")
    print("=" * 75)

    orch = OrchestratorAgent()
    results = run_benchmark(orch, rounds=3)
    print_results(results)

    output_path = Path(__file__).resolve().parent.parent / "data" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
