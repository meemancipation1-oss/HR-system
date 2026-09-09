"""LLM-as-Judge 评估脚本 — 评估简历筛选准确率。

Run:
  python main.py evaluate              # 每个场景 5 个案例
  python main.py evaluate --cases 2    # 每个场景 2 个案例 (快速)
  python main.py evaluate --offline    # 只构建黄金集, 不调用 LLM
  python main.py evaluate --seed 7     # 固定随机种子, 可复现
  python main.py evaluate --sleep 3    # 案例之间间隔 3 秒 (限流时用)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.evaluator import run_evaluation, print_report


def main() -> None:
    args = sys.argv[1:]

    n_cases = 5
    seed = 42
    offline = False
    sleep_between = 0.0
    judge_model: str | None = None
    gate = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--cases" and i + 1 < len(args):
            n_cases = max(1, int(args[i + 1]))
            i += 2
        elif arg == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif arg == "--judge-model" and i + 1 < len(args):
            judge_model = args[i + 1]
            i += 2
        elif arg == "--sleep" and i + 1 < len(args):
            sleep_between = max(0.0, float(args[i + 1]))
            i += 2
        elif arg == "--offline":
            offline = True
            i += 1
        elif arg == "--gate":
            gate = True
            i += 1
        else:
            print(f"未知参数: {arg}")
            print(__doc__)
            return

    if offline:
        print("运行模式: --offline (仅黄金集, 不调用 LLM)\n")
        if gate:
            raise SystemExit("--offline 不能与 --gate 同时使用；离线模式不代表 Agent 评测通过。")

    _, report = run_evaluation(
        n_per_scenario=n_cases,
        judge_model=judge_model,
        offline=offline,
        seed=seed,
        sleep_between=sleep_between,
    )
    print_report(report)

    # 路由层评估 (确定性规则, 无需 LLM) — 与筛选评测一起输出
    from core.routing_eval import print_routing_report, run_routing_evaluation
    routing_metrics = run_routing_evaluation()
    print_routing_report(routing_metrics)
    out_path = Path(__file__).resolve().parent.parent / "data" / "routing_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(routing_metrics, f, ensure_ascii=False, indent=2)
    print(f"      路由评估已保存: {out_path}")

    if gate:
        from scripts.regression_gate import check_report
        from pathlib import Path
        baseline_path = Path(__file__).resolve().parent.parent / "data" / "eval_baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        failures = check_report(report, baseline, routing_metrics)
        if failures:
            print("\nREGRESSION DETECTED")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(1)
        print("\nREGRESSION GATE: PASS")


if __name__ == "__main__":
    main()

