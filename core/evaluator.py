"""LLM-as-Judge 评估模块 — 评估简历筛选 (Screening) 的准确率。

核心思想:
  1. 黄金测试集: 用确定性规则从 CSV 数据构建 ground truth (真实匹配标签 + 规则打分)
  2. 被测系统:   让 ScreeningAgent 对同样的 候选人+岗位要求 执行筛选评分
  3. Judge LLM:  独立的大模型参照真实标签, 判定 Agent 的筛选结论是否正确
  4. 汇总指标:   准确率 / 精确率 / 召回率 / F1 / 评分MAE / Judge一致率

用法 (CLI):
  python main.py evaluate            # 每个场景 5 个案例, 完整流程
  python main.py evaluate --cases 2  # 每个场景 2 个案例, 快速验证
  python main.py evaluate --offline  # 只构建黄金集 + 规则指标, 不调用 LLM
"""

from __future__ import annotations
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pydantic import BaseModel, Field
from config.settings import HR_DATA_DIR
from core.llm import structured_completion

# 判定为"匹配"的分数阈值 (规则评分 与 Agent 评分 共用)
MATCH_THRESHOLD = 60.0

# 筛选失败时的重试策略 (应对 API 限流/瞬时错误)
MAX_SCREEN_ATTEMPTS = 3
SCREEN_RETRY_BACKOFF = 5   # 秒, 第 n 次重试等待 = backoff * n

# ═══════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════

class EvalCase(BaseModel):
    """一条评估用例: 候选人 + 岗位要求 + 规则生成的真实标签。"""

    case_id: str
    person_id: str
    姓名: str
    岗位名称: str
    岗位要求: str
    真实标签: str                    # "匹配" / "不匹配"
    真实评分: float                  # 规则打分 0-100
    规则依据: str                    # 每条规则的命中明细, 供 Judge 参考


class ScreeningEvalResult(BaseModel):
    """单个案例的完整评估结果。"""

    case_id: str
    person_id: str
    姓名: str
    岗位名称: str
    真实标签: str
    真实评分: float
    agent_匹配分数: float            # ScreeningAgent 输出的分数
    agent_标签: str                  # 按阈值折算: "匹配" / "不匹配"
    judge_结论: str = ""             # Judge 判定: 正确 / 部分正确 / 错误 (由 Judge 阶段填充)
    judge_一致程度: float = 0.0      # Judge 打分 0-100 (由 Judge 阶段填充)
    judge_误判原因: str = ""
    judge_改进建议: str = ""
    agent_输出原文: str = ""          # Agent 完整输出, 便于人工复核
    error: str = ""                  # 若执行失败, 记录错误信息


class JudgeVerdict(BaseModel):
    """Judge LLM 的结构化判定输出。"""

    结论: str = Field(description="判定结论, 只能是: 正确 / 部分正确 / 错误")
    一致程度: float = Field(description="Agent 筛选结论与真实标签的一致程度(0-100)", ge=0, le=100)
    误判原因: str = Field(description="若结论不是完全正确, 分析误判的可能原因")
    改进建议: str = Field(description="针对本案例给 ScreeningAgent 的可操作改进建议")


# ═══════════════════════════════════════
#  黄金测试集构建 — 基于确定性规则
# ═══════════════════════════════════════

# 规则: (列名, 操作, 值, 权重)
# 操作说明:
#   ==            字段值等于 值
#   in            字段值属于 值(list)
#   not_empty     字段非空
#   contains_any  字段文本包含 值(list) 中任意一个关键词
#   le            数值 <= 值
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "高级技术研发岗",
        "requirements": "硕士及以上学历；有发明专利；有论文著述；专业技能需涵盖人工智能或计算机方向",
        "rules": [
            ("学历", "in", ["硕士", "博士"], 30),
            ("发明专利", "not_empty", None, 25),
            ("论文著述", "not_empty", None, 25),
            ("专业技能", "contains_any", ["人工智能", "计算机", "软件", "算法"], 20),
        ],
    },
    {
        "name": "科研项目管理岗",
        "requirements": "硕士及以上学历；具备职业资格；有管理经验；有论文著述；年龄45岁以下",
        "rules": [
            ("学历", "in", ["硕士", "博士"], 30),
            ("职业资格", "not_empty", None, 20),
            ("管理经验", "not_empty", None, 20),
            ("论文著述", "not_empty", None, 15),
            ("年龄", "le", 45, 15),
        ],
    },
    {
        "name": "信息化建设岗",
        "requirements": "本科及以上学历；专业技能涵盖计算机/软件/数据方向；年龄40岁以下；有发明专利或软件著作权",
        "rules": [
            ("学历", "in", ["本科", "硕士", "博士"], 30),
            ("专业技能", "contains_any", ["计算机", "软件", "信息", "网络", "数据"], 30),
            ("年龄", "le", 40, 20),
            ("发明专利", "not_empty", None, 10),
            ("软件著作权", "not_empty", None, 10),
        ],
    },
    {
        "name": "基层指挥管理岗",
        "requirements": "本科及以上学历；有管理经验；有军衔等级；年龄45岁以下；沟通协调能力良好",
        "rules": [
            ("学历", "in", ["本科", "硕士", "博士"], 25),
            ("管理经验", "in", ["丰富", "中等"], 25),
            ("沟通协调能力", "ge", 7, 20),
            ("年龄", "le", 40, 15),
            ("军衔等级", "in", ["少校", "中校", "上校", "大校", "少将"], 15),
        ],
    },
]


class RuleLabeler:
    """基于确定性规则的 ground-truth 标签器 (可复现、无需 LLM、向量化打分)。"""

    def __init__(self) -> None:
        self._df = pd.read_csv(
            HR_DATA_DIR / "SyntheticIndividuals.csv",
            encoding="utf-8",
            dtype=str,
            low_memory=False,
        )

    @staticmethod
    def _mask(rows: pd.DataFrame, field: str, op: str, value: Any) -> pd.Series:
        """对整批数据生成单条规则的布尔掩码 (向量化, 避免逐行循环)。"""
        if field not in rows.columns:
            return pd.Series(False, index=rows.index)
        raw = rows[field].fillna("").astype(str).str.strip()

        if op == "not_empty":
            return raw != ""
        if op == "==":
            return raw == str(value)
        if op == "in":
            return raw.isin({str(v) for v in (value or [])})
        if op == "contains_any":
            pattern = "|".join(re.escape(k) for k in (value or []))
            return raw.str.contains(pattern, regex=True, na=False)
        if op in ("le", "ge"):
            numeric = pd.to_numeric(raw, errors="coerce")
            return numeric.le(float(value)) if op == "le" else numeric.ge(float(value))
        return pd.Series(False, index=rows.index)

    def _score_batch(self, rows: pd.DataFrame, rules: list[tuple]) -> pd.Series:
        """批量计算规则得分 (0-100)。"""
        total = sum(w for _, _, _, w in rules)
        score = pd.Series(0.0, index=rows.index)
        for field, op, value, weight in rules:
            score += self._mask(rows, field, op, value).astype(float) * weight
        return (score / total * 100).round(1)

    def _basis_text(self, row: pd.Series, rules: list[tuple]) -> str:
        """单条候选人的规则命中明细 (只对最终选中的少量样本执行)。

        示例: 学历=博士(命中+40, 实际: 博士) | 年龄≤40(未命中, 实际: 45)"""
        hits: list[str] = []
        for field, op, value, weight in rules:
            ok = bool(self._mask(row.to_frame().T, field, op, value).iloc[0])
            actual = row.get(field, "")
            if pd.isna(actual):
                actual = ""
            actual = str(actual).strip() or "(空)"
            if op == "not_empty":
                rule_desc = f"{field}非空"
            elif op == "contains_any":
                rule_desc = f"{field}包含{value}"
            elif op == "in":
                rule_desc = f"{field}∈{value}"
            elif op == "le":
                rule_desc = f"{field}≤{value}"
            elif op == "ge":
                rule_desc = f"{field}≥{value}"
            else:
                rule_desc = f"{field}={value}"
            mark = f"命中+{weight}" if ok else "未命中"
            hits.append(f"{rule_desc}({mark}, 实际: {actual})")
        return " | ".join(hits)

    def build_cases(self, n_per_scenario: int = 5, seed: int = 42) -> list[EvalCase]:
        """为每个场景采样候选人, 生成均衡的正负样本测试集。"""
        rng = random.Random(seed)
        cases: list[EvalCase] = []

        for scenario_idx, scenario in enumerate(SCENARIOS):
            name = scenario["name"]
            requirements = scenario["requirements"]
            rules = scenario["rules"]

            # 每场景采样 2 万行, 向量化打分后按 匹配/不匹配 均衡抽取
            pool = self._df.sample(n=min(20000, len(self._df)), random_state=seed + scenario_idx)
            scores = self._score_batch(pool, rules)
            labels = scores.ge(MATCH_THRESHOLD)

            matched_idx = pool.index[labels]
            unmatched_idx = pool.index[~labels]
            half = (n_per_scenario + 1) // 2
            chosen_idx = list(matched_idx[:half]) + list(unmatched_idx[: n_per_scenario - half])
            rng.shuffle(chosen_idx)

            for i, idx in enumerate(chosen_idx):
                row = pool.loc[idx]
                score = float(scores.loc[idx])
                label = "匹配" if score >= MATCH_THRESHOLD else "不匹配"
                cases.append(
                    EvalCase(
                        case_id=f"{name}-{i + 1}",
                        person_id=str(row["人员ID"]).strip(),
                        姓名=str(row["姓名"]).strip(),
                        岗位名称=name,
                        岗位要求=requirements,
                        真实标签=label,
                        真实评分=score,
                        规则依据=self._basis_text(row, rules),
                    )
                )

        return cases

    @staticmethod
    def fact_summary(case: EvalCase) -> str:
        """从主表抽取候选人关键事实, 供 Judge 参照 (只取规则涉及字段)。"""
        df = pd.read_csv(
            HR_DATA_DIR / "SyntheticIndividuals.csv",
            encoding="utf-8",
            dtype=str,
            low_memory=False,
        )
        row = df[df["人员ID"] == case.person_id]
        if row.empty:
            return "未找到候选人档案"
        r = row.iloc[0]
        fields = ["姓名", "性别", "年龄", "学历", "军衔等级", "职务类别", "职业资格",
                  "专业技能", "擅长领域", "管理经验", "沟通协调能力", "论文著述",
                  "发明专利", "软件著作权"]
        parts = []
        for f in fields:
            v = r.get(f, "")
            if pd.notna(v) and str(v).strip():
                parts.append(f"{f}: {v}")
        return "\n".join(parts)


# ═══════════════════════════════════════
#  LLM-as-Judge
# ═══════════════════════════════════════

class LLMJudge:
    """独立的 Judge LLM — 参照真实标签判定 Agent 筛选结论的准确性。

    设计要点:
      - Judge 与 ScreeningAgent 使用不同的模型时 (JUDGE_MODEL 环境变量),
        可避免"自评自"的偏好偏差 (self-preference bias)
      - 输出通过 structured_completion 强制为 JudgeVerdict 结构
    """

    SYSTEM_PROMPT = """你是一个严格的 AI 评估员 (LLM-as-Judge)，负责评估 HR 筛选 Agent 的工作质量。

你的任务:
1. 参照"真实标签"(由确定性规则生成) 判断 Agent 的筛选结论是否准确
2. 真实标签只反映规则命中的客观事实，Agent 可能给出更全面的评估
3. 若 Agent 结论与真实标签一致（或 Agent 提供了合理且更细致的判断），应判为"正确"
4. 若 Agent 忽略了关键事实或结论与事实相悖，判为"部分正确"或"错误"
5. 请用中文客观输出，理由要引用具体事实"""

    def __init__(self, judge_model: str | None = None) -> None:
        self.judge_model = judge_model

    def judge(self, case: EvalCase, agent_output: str) -> JudgeVerdict:
        """对单个案例的 Agent 输出做出结构化判定。"""
        facts = RuleLabeler.fact_summary(case)
        prompt = f"""请评估以下 HR 筛选 Agent 的工作质量。

## 岗位要求
{case.岗位名称}: {case.岗位要求}

## 候选人事实摘要 (来自档案数据库)
{facts}

## 真实标签 (由确定性规则生成, 参考标准)
- 真实评分: {case.真实评分}/100
- 真实标签: {case.真实标签}
- 规则依据: {case.规则依据}

## ScreeningAgent 的筛选输出
{agent_output}

请判定 Agent 的筛选结论是否准确：
1. Agent 是否基于候选人事实做出判断（而非凭空捏造）？
2. Agent 的评分/结论与真实标签是否一致？差异是否合理？
3. 是否存在关键事实遗漏或明显误判？"""
        verdict = structured_completion(
            messages=[{"role": "user", "content": prompt}],
            output_model=JudgeVerdict,
            model=self.judge_model,
            system_prompt=self.SYSTEM_PROMPT,
        )
        return verdict


# ═══════════════════════════════════════
#  指标计算
# ═══════════════════════════════════════

def compute_metrics(results: list[ScreeningEvalResult]) -> dict[str, Any]:
    """基于规则标签计算分类指标 + Judge 一致率。"""
    n = len(results)
    if n == 0:
        return {}

    tp = fp = fn = tn = 0
    score_diffs: list[float] = []
    judge_correct = 0
    judge_partial = 0
    judge_wrong = 0
    judge_acceptable = 0

    for r in results:
        pred = r.agent_标签 == "匹配"
        truth = r.真实标签 == "匹配"
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1

        score_diffs.append(abs(r.agent_匹配分数 - r.真实评分))
        if r.judge_结论 == "正确":
            judge_correct += 1
        elif r.judge_结论 == "部分正确":
            judge_partial += 1
        elif r.judge_结论 == "错误":
            judge_wrong += 1
        if r.judge_结论 in ("正确", "部分正确"):
            judge_acceptable += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n

    near10 = sum(1 for d in score_diffs if d <= 10) / len(score_diffs) if score_diffs else None
    near15 = sum(1 for d in score_diffs if d <= 15) / len(score_diffs) if score_diffs else None
    return {
        "案例总数": n,
        "准确率": round(accuracy, 4),
        "精确率": round(precision, 4),
        "召回率": round(recall, 4),
        "F1": round(f1, 4),
        "评分MAE": round(sum(score_diffs) / len(score_diffs), 2) if score_diffs else None,
        "评分误差≤10占比": round(near10, 4) if near10 is not None else None,
        "评分误差≤15占比": round(near15, 4) if near15 is not None else None,
        "Judge正确数": judge_correct,
        "Judge部分正确数": judge_partial,
        "Judge错误数": judge_wrong,
        "Judge一致率": round(judge_correct / n, 4),
        "Judge认可率": round(judge_acceptable / n, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def build_report(results: list[ScreeningEvalResult]) -> dict[str, Any]:
    """汇总总体指标 + 按场景拆分。"""
    report: dict[str, Any] = {
        "生成时间": None,
        "阈值": MATCH_THRESHOLD,
        "总体指标": compute_metrics(results),
        "按场景": {},
        "案例明细": [r.model_dump() for r in results],
    }
    scenes = sorted({r.岗位名称 for r in results})
    for scene in scenes:
        scene_results = [r for r in results if r.岗位名称 == scene]
        report["按场景"][scene] = compute_metrics(scene_results)
    return report


# ═══════════════════════════════════════
#  主流程
# ═══════════════════════════════════════

def run_evaluation(
    n_per_scenario: int = 5,
    judge_model: str | None = None,
    offline: bool = False,
    seed: int = 42,
    sleep_between: float = 0.0,
) -> tuple[list[ScreeningEvalResult], dict[str, Any]]:
    """完整评估流程: 构建黄金集 → ScreeningAgent 筛选 → Judge 判定 → 指标。"""
    from agents.screening_agent import ScreeningAgent, ScreeningOutput
    import json as _json

    print("=" * 70)
    print("  LLM-as-Judge 评估 — 简历筛选准确率")
    print("=" * 70)

    # 1. 黄金测试集
    print(f"\n[1/4] 构建黄金测试集 (规则 ground truth)...")
    labeler = RuleLabeler()
    cases = labeler.build_cases(n_per_scenario=n_per_scenario, seed=seed)
    print(f"      共 {len(cases)} 个案例, "
          f"匹配 {sum(1 for c in cases if c.真实标签 == '匹配')} 个, "
          f"不匹配 {sum(1 for c in cases if c.真实标签 == '不匹配')} 个")

    # 保存黄金集 (可复现)
    cases_path = Path(__file__).resolve().parent.parent / "data" / "eval_cases.json"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cases_path, "w", encoding="utf-8") as f:
        f.write(_json.dumps([c.model_dump() for c in cases], ensure_ascii=False, indent=2))
    print(f"      黄金集已保存: {cases_path}")

    if offline:
        print("\n[--offline] 跳过 LLM 调用, 只输出黄金集统计。")
        fake = [
            ScreeningEvalResult(
                case_id=c.case_id, person_id=c.person_id, 姓名=c.姓名,
                岗位名称=c.岗位名称, 真实标签=c.真实标签, 真实评分=c.真实评分,
                agent_匹配分数=c.真实评分, agent_标签=c.真实标签,
                judge_结论="正确", judge_一致程度=100.0,
            )
            for c in cases
        ]
        return fake, build_report(fake)

    # 2. ScreeningAgent 筛选 (失败自动重试, 应对 API 限流/瞬时错误)
    # 注意: 每个案例必须用全新 Agent 实例, 否则 self.memory 会跨案例污染上下文
    print(f"\n[2/4] 运行 ScreeningAgent 筛选 {len(cases)} 个案例...")
    results: list[ScreeningEvalResult] = []
    for idx, case in enumerate(cases, start=1):
        print(f"      [{idx}/{len(cases)}] {case.姓名}({case.person_id}) x {case.岗位名称}...", end=" ")
        raw: str | None = None
        last_error = ""
        agent_score = -1.0
        agent_label = "失败"
        agent_text = ""

        for attempt in range(1, MAX_SCREEN_ATTEMPTS + 1):
            try:
                agent = ScreeningAgent()  # 每个案例独立实例, 隔离对话记忆
                raw = agent.screen_candidate(case.person_id, requirements=case.岗位要求)
                # screen_candidate 返回 ScreeningOutput 的 JSON 字符串
                if isinstance(raw, str):
                    output = ScreeningOutput(**_json.loads(raw))
                else:
                    output = raw
                agent_score = float(output.匹配分数)
                agent_label = "匹配" if agent_score >= MATCH_THRESHOLD else "不匹配"
                agent_text = output.model_dump_json(ensure_ascii=False, indent=2)
                retry_note = f" (重试{attempt - 1}次)" if attempt > 1 else ""
                print(f"✓ 分数={agent_score:.0f} ({agent_label}){retry_note}")
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e} | 原始返回: {str(raw)[:150] if raw else '(空/非JSON)'}"
                if attempt < MAX_SCREEN_ATTEMPTS:
                    wait = SCREEN_RETRY_BACKOFF * attempt
                    print(f"重试{attempt}/{MAX_SCREEN_ATTEMPTS} ({wait}s后)...", end=" ")
                    time.sleep(wait)
        else:
            print(f"✗ 失败: {last_error}")
            results.append(
                ScreeningEvalResult(
                    case_id=case.case_id, person_id=case.person_id, 姓名=case.姓名,
                    岗位名称=case.岗位名称, 真实标签=case.真实标签, 真实评分=case.真实评分,
                    agent_匹配分数=agent_score, agent_标签=agent_label,
                    judge_结论="错误", judge_一致程度=0.0,
                    error=last_error,
                )
            )
            if sleep_between:
                time.sleep(sleep_between)
            continue

        results.append(
            ScreeningEvalResult(
                case_id=case.case_id, person_id=case.person_id, 姓名=case.姓名,
                岗位名称=case.岗位名称, 真实标签=case.真实标签, 真实评分=case.真实评分,
                agent_匹配分数=agent_score, agent_标签=agent_label,
                agent_输出原文=agent_text,
            )
        )
        if sleep_between:
            time.sleep(sleep_between)

    # 3. Judge LLM 判定 (失败自动重试)
    print(f"\n[3/4] LLM-as-Judge 判定 {len(results)} 个案例...")
    judge = LLMJudge(judge_model=judge_model)
    for idx, r in enumerate(results, start=1):
        if r.error:
            continue
        case = next(c for c in cases if c.case_id == r.case_id)
        print(f"      [{idx}/{len(results)}] Judge: {r.姓名} x {r.岗位名称}...", end=" ")
        last_error = ""
        for attempt in range(1, MAX_SCREEN_ATTEMPTS + 1):
            try:
                verdict = judge.judge(case, r.agent_输出原文)
                r.judge_结论 = verdict.结论
                r.judge_一致程度 = verdict.一致程度
                r.judge_误判原因 = verdict.误判原因
                r.judge_改进建议 = verdict.改进建议
                retry_note = f" (重试{attempt - 1}次)" if attempt > 1 else ""
                print(f"✓ {verdict.结论} (一致程度 {verdict.一致程度:.0f}){retry_note}")
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < MAX_SCREEN_ATTEMPTS:
                    wait = SCREEN_RETRY_BACKOFF * attempt
                    print(f"重试{attempt}/{MAX_SCREEN_ATTEMPTS} ({wait}s后)...", end=" ")
                    time.sleep(wait)
        else:
            r.judge_结论 = "错误"
            r.judge_一致程度 = 0.0
            r.judge_误判原因 = f"Judge 调用失败: {last_error}"
            print(f"✗ Judge 失败: {last_error}")

    # 4. 指标汇总
    print(f"\n[4/4] 生成评估报告...")
    from datetime import datetime
    report = build_report(results)
    report["生成时间"] = datetime.now().isoformat(timespec="seconds")

    out_path = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_json.dumps(report, ensure_ascii=False, indent=2))
    print(f"      报告已保存: {out_path}")

    return results, report


def print_report(report: dict[str, Any]) -> None:
    """终端友好输出评估报告。"""
    print("\n" + "=" * 70)
    print("  📊 LLM-as-Judge 评估报告 — 简历筛选准确率")
    print("=" * 70)

    m = report.get("总体指标", {})
    print(f"\n  ⚖️  匹配阈值: {report.get('阈值', MATCH_THRESHOLD)} 分")
    print(f"\n  📈 总体指标:")
    for k in ["案例总数", "准确率", "精确率", "召回率", "F1", "评分MAE",
              "评分误差≤10占比", "评分误差≤15占比",
              "Judge正确数", "Judge部分正确数", "Judge错误数",
              "Judge一致率", "Judge认可率"]:
        v = m.get(k)
        if isinstance(v, float):
            if k in ("准确率", "精确率", "召回率", "评分误差≤10占比", "评分误差≤15占比",
                     "Judge一致率", "Judge认可率"):
                v = f"{v:.2%}"
            else:
                v = f"{v:.2f}"
        print(f"    {k:<12}: {v}")
    print(f"    {'混淆矩阵':<12}: TP={m.get('TP')} FP={m.get('FP')} FN={m.get('FN')} TN={m.get('TN')}")
    print(f"    {'口径说明':<12}: 认可率=(正确+部分正确)/总数; 一致率=仅正确/总数; 误差占比=评分绝对差≤阈值")

    scenes = report.get("按场景", {})
    if scenes:
        print(f"\n  🧩 按场景:")
        for name, sm in scenes.items():
            print(f"    {name}: 准确率={sm.get('准确率', 0):.2%} "
                  f"F1={sm.get('F1', 0):.2f} MAE={sm.get('评分MAE', '—')} "
                  f"Judge一致率={sm.get('Judge一致率', 0):.2%} 认可率={sm.get('Judge认可率', 0):.2%}")

    # 误判案例 Top 展示
    details = report.get("案例明细", [])
    errors = [d for d in details if d.get("judge_结论") in ("部分正确", "错误") or d.get("error")]
    if errors:
        print(f"\n  ⚠️  待改进案例 ({len(errors)} 个):")
        for d in errors[:8]:
            print(f"    - {d.get('姓名')}({d.get('person_id')}) x {d.get('岗位名称')}: "
                  f"真实={d.get('真实标签')}({d.get('真实评分')}) "
                  f"Agent={d.get('agent_标签')}({d.get('agent_匹配分数')}) "
                  f"Judge={d.get('judge_结论')}")
    print("=" * 70)


__all__ = [
    "EvalCase",
    "ScreeningEvalResult",
    "JudgeVerdict",
    "RuleLabeler",
    "LLMJudge",
    "SCENARIOS",
    "MATCH_THRESHOLD",
    "compute_metrics",
    "build_report",
    "run_evaluation",
    "print_report",
]












