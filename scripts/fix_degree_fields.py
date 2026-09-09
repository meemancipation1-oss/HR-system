"""修复 SyntheticIndividuals.csv 中学历字段自相矛盾的问题。

背景: 合成数据把"本科学位/硕士学位/博士学位 + 各层级毕业院校/专业"对所有人
全部填满, 与权威字段 `学历` 矛盾, 导致 ScreeningAgent 误读出更高学历
(学历通胀)。抽查 20 例评测中约 11 例命中。

修复规则: `学历` 是唯一权威; 清空**高于**该学历级别的 学位/毕业院校/专业 字段。
- 学历=大专 -> 清空 本科/硕士/博士 三级字段
- 学历=本科 -> 保留本科, 清空 硕士/博士
- 学历=硕士 -> 保留本科/硕士, 清空 博士
- 学历=博士 -> 全部保留 (正常学历递进)

Run:
  python scripts/fix_degree_fields.py
"""
from __future__ import annotations
import shutil
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "SynData_1w"
CSV = DATA_DIR / "SyntheticIndividuals.csv"

# 学历层级 (值必须是 CSV 中 `学历` 字段的实际取值)
LEVELS = ["大专", "本科", "硕士", "博士"]
# 每个层级需要清理的字段
LEVEL_FIELDS = {
    "本科": ["本科学位", "本科毕业院校", "本科专业"],
    "硕士": ["硕士学位", "硕士毕业院校", "硕士专业"],
    "博士": ["博士学位", "博士毕业院校", "博士专业"],
}


def main() -> None:
    if not CSV.exists():
        raise FileNotFoundError(f"未找到数据文件: {CSV}")

    df = pd.read_csv(CSV, encoding="utf-8", dtype=str, low_memory=False)
    rank = {lv: i for i, lv in enumerate(LEVELS)}
    before = df.copy()
    changed = 0

    for idx, row in df.iterrows():
        edu = str(row.get("学历", "")).strip()
        edu_rank = rank.get(edu, -1)  # 未知值按最低处理
        for lv, cols in LEVEL_FIELDS.items():
            if rank[lv] > edu_rank:
                for col in cols:
                    if pd.notna(row.get(col)) and str(row[col]).strip():
                        df.at[idx, col] = ""
                        changed += 1

    if changed == 0:
        print("无矛盾字段, 无需修复。")
        return

    # 备份后写入
    backup = CSV.with_name(CSV.name + ".bak")
    if not backup.exists():
        shutil.copy2(CSV, backup)
        print(f"已备份原始文件: {backup}")
    df.to_csv(CSV, index=False, encoding="utf-8")

    # 修复后自检
    check = pd.read_csv(CSV, encoding="utf-8", dtype=str, low_memory=False)
    bad = len(check[(check["学历"] != "博士") & (check["博士学位"] == "博士")])
    print(f"已清理 {changed} 个矛盾字段 (影响 {len(before) - len(after_diff(before, df))} 行)")
    print(f"修复后残留矛盾行数: {bad} (应为 0)")


def after_diff(before, after):
    return after[~after.eq(before).all(axis=1)]


if __name__ == "__main__":
    main()
