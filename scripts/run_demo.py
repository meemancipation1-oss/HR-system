"""CLI demo for the HR Multi-Agent system — quick showcase without UI."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator_agent import OrchestratorAgent


orch = OrchestratorAgent()


# ── Demo Cases ──

def _print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demo_1_explore():
    """Demo: Explore the dataset."""
    _print_header("Demo 1: 数据探索 — 查看数据集概览")
    result = orch.explore_dataset()
    print(result)


def demo_2_semantic_search():
    """Demo: Semantic search for candidates."""
    _print_header("Demo 2: 语义搜索 — 寻找有博士学历且有发明专利的候选人")
    results = orch.search_semantic("博士学历 发明专利", k=3)
    for r in results:
        print(f"  👤 {r['姓名']} ({r['人员ID']}) — {r['摘要'][:100]}...")


def demo_3_screening():
    """Demo: Screen a candidate."""
    _print_header("Demo 3: 候选人筛选 — 评估人员 1 匹配高级技术岗位")
    result = orch.run_pipeline(
        "screening_agent",
        "请评估人员ID=1 的候选人是否适合高级技术研发岗位。要求：博士学历、有发明专利、科研项目经验。",
    )
    print(result)


def demo_4_analysis():
    """Demo: Deep analysis of a candidate."""
    _print_header("Demo 4: 深度分析 — 分析人员 1 的能力画像")
    result = orch.run_pipeline(
        "analysis_agent",
        "请对人员ID=1 进行深度能力分析。",
    )
    print(result)


def demo_5_interview():
    """Demo: Generate interview questions."""
    _print_header("Demo 5: 面试问题生成 — 为人员 1 生成面试问题")
    result = orch.run_pipeline(
        "interview_agent",
        "请为人员ID=1 生成5个针对性面试问题，目标岗位为高级技术管理岗。",
    )
    print(result)


def demo_6_report():
    """Demo: Generate recruitment report."""
    _print_header("Demo 6: 招聘报告 — 为技术总监岗位生成报告")
    result = orch.run_pipeline(
        "report_agent",
        "请为岗位【技术总监】生成招聘报告。候选人为人员ID=1,2,3",
    )
    print(result)


def interactive_mode():
    """Interactive CLI chat mode."""
    print("\n💬 进入交互模式（输入 exit 退出）")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                break
            print("\nAgent 思考中...")
            response = orch.run(user_input)
            print(f"\nAI Agent:\n{response}")
        except KeyboardInterrupt:
            break


# ── Main ──

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║     🤖 HR Multi-Agent Recruitment System             ║")
    print("║     Native Tool Calling + RAG + DeepSeek/OpenAI      ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n选择演示模式:")
    print("  1. 快速演示所有功能")
    print("  2. 数据探索")
    print("  3. 语义搜索")
    print("  4. 候选人筛选")
    print("  5. 深度分析")
    print("  6. 面试问题生成")
    print("  7. 招聘报告生成")
    print("  8. 交互模式")
    print("  0. 退出")

    choice = input("\n请选择 (0-8): ").strip()

    demos = {
        "1": [demo_1_explore, demo_2_semantic_search, demo_3_screening,
              demo_4_analysis, demo_5_interview, demo_6_report],
        "2": [demo_1_explore],
        "3": [demo_2_semantic_search],
        "4": [demo_3_screening],
        "5": [demo_4_analysis],
        "6": [demo_5_interview],
        "7": [demo_6_report],
        "8": [interactive_mode],
    }

    if choice in demos:
        for demo in demos[choice]:
            demo()
        if choice != "8":
            print("\n✅ 演示完成！")
    elif choice == "0":
        pass
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
