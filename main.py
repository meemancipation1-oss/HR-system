"""HR Multi-Agent Recruitment System — Entry Point.

Usage:
    # Interactive CLI demo
    python main.py demo

    # Start API server
    python main.py api

    # Build vector index
    python main.py ingest

    # LLM-as-Judge evaluation
    python main.py evaluate [--cases N] [--offline] [--seed N] [--judge-model M]

    # Explain the planner-first route for one query
    python main.py route 不要博士学历的候选人

    # Streamlit UI
    streamlit run ui/app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "demo":
        from scripts.run_demo import main as demo_main
        demo_main()
    elif cmd == "api":
        from scripts.run_api import main as api_main
        api_main()
    elif cmd == "ingest":
        from scripts.ingest_data import main as ingest_main
        ingest_main()
    elif cmd == "evaluate":
        from scripts.evaluate import main as eval_main
        # 剥离命令名, 只把 --xxx 参数传给评估脚本
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        eval_main()
    elif cmd == "route":
        from scripts.route_check import main as route_main
        route_main(" ".join(sys.argv[2:]))
    elif cmd == "ui":
        print("Run: streamlit run ui/app.py")
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
