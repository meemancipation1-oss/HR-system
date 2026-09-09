"""Explain the planner-first route for one natural-language chat request.

Run:
  python main.py route 不要本科生
"""
from __future__ import annotations

import sys


def main(query: str = "") -> None:
    query = query or " ".join(sys.argv[1:])
    if not query:
        print(__doc__)
        return
    print(f"输入: {query}")
    print("路由: 任务规划器 -> 结构化 HiringTaskPlan -> 按 actions 调用领域 Agent")
    print("说明: 聊天请求不走正则快路；否定条件由规划器写入 exclusions。")


if __name__ == "__main__":
    main()
