"""Regression quality gate for routing and screening evaluation reports.

Usage:
    python scripts/regression_gate.py
    python main.py evaluate --gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.routing_eval import run_routing_evaluation


def check_report(report: dict, baseline: dict, routing: dict) -> list[str]:
    thresholds = baseline["thresholds"]
    metrics = report.get("总体指标", {})
    failures: list[str] = []
    for metric in ("准确率", "F1"):
        actual = metrics.get(metric)
        minimum = thresholds[f"{metric}_min"]
        if actual is None or actual < minimum:
            failures.append(f"{metric}: actual={actual}, required>={minimum}")
    mae = metrics.get("评分MAE")
    if mae is None or mae > thresholds["评分MAE_max"]:
        failures.append(f"评分MAE: actual={mae}, required<={thresholds['评分MAE_max']}")
    routing_accuracy = routing["总准确率"]
    if routing_accuracy < thresholds["路由准确率_min"]:
        failures.append(f"路由准确率: actual={routing_accuracy}, required>={thresholds['路由准确率_min']}")
    return failures


def main(report_path: Path | None = None) -> int:
    report_path = report_path or ROOT / "data" / "eval_results.json"
    baseline_path = ROOT / "data" / "eval_baseline.json"
    if not report_path.exists():
        print(f"缺少评估报告: {report_path}")
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    routing = run_routing_evaluation()
    failures = check_report(report, baseline, routing)
    print(f"基线版本: {baseline['version']}")
    print(f"筛选指标: Accuracy={report['总体指标'].get('准确率')}, "
          f"F1={report['总体指标'].get('F1')}, MAE={report['总体指标'].get('评分MAE')}")
    print(f"路由指标: Accuracy={routing['总准确率']}")
    if failures:
        print("REGRESSION DETECTED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("REGRESSION GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
