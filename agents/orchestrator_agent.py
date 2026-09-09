"""Orchestrator Agent — plan first, then execute domain-agent actions."""
#负责决定用户的问题应该交给哪个 Agent 去处理。分三级路由：
from __future__ import annotations
import json
import time
from enum import Enum

from agents.data_agent import DataAgent
from agents.screening_agent import ScreeningAgent
from agents.analysis_agent import AnalysisAgent
from agents.interview_agent import InterviewAgent
from agents.report_agent import ReportAgent
from core.llm import structured_completion
from core.memory import AgentMemory
from core.observability import configure_logging, log_event
from core.runtime import RuntimeFailure
from config.settings import LOG_PATH
from models.task_plan import HiringTaskPlan, PLANNER_SYSTEM_PROMPT

_LOGGER = configure_logging(LOG_PATH)


def _failure_json(exc: Exception, agent: str) -> str:
    name = type(exc).__name__.lower()
    timeout = "timeout" in name
    failure = RuntimeFailure(
        status="incomplete" if timeout else "error",
        error_code="llm_timeout" if timeout else "llm_unavailable",
        message="模型调用超时，请稍后重试。" if timeout else "模型服务暂时不可用。",
        retryable=True,
        details={"agent": agent, "error_type": type(exc).__name__},
    )
    return failure.model_dump_json(ensure_ascii=False)

#定义的是任务类型枚举（Enum）。
class TaskType(str, Enum):
    EXPLORE = "explore_dataset"
    SEMANTIC_SEARCH = "semantic_search"
    SCREEN = "screen_candidate"
    ANALYZE = "analyze_candidate"
    INTERVIEW = "generate_interview"
    REPORT = "generate_report"
    RANK = "rank_candidates"
    COMPARE = "compare_candidates"
    AUTO = "auto"


class OrchestratorAgent:
    """Master orchestrator that routes to the correct sub-agent."""

    def __init__(self, messages: list[dict] | None = None) -> None:   #初始化
        self._memory = AgentMemory(messages)
        self._agents = { #建立了Agent池
            "data_agent": DataAgent(),
            "screening_agent": ScreeningAgent(),
            "analysis_agent": AnalysisAgent(),
            "interview_agent": InterviewAgent(),
            "report_agent": ReportAgent(),
        }
        self._task_to_agent = {  #就是映射。
            TaskType.EXPLORE: "data_agent",
            TaskType.SEMANTIC_SEARCH: "data_agent",
            TaskType.SCREEN: "screening_agent",
            TaskType.ANALYZE: "analysis_agent",
            TaskType.INTERVIEW: "interview_agent",
            TaskType.REPORT: "report_agent",
            TaskType.RANK: "report_agent",
            TaskType.COMPARE: "report_agent",
            TaskType.AUTO: None,
        }

    def get_memory_snapshot(self) -> list[dict]:
        """Return conversation state for the API session store."""
        return self._memory.snapshot()

    def _sync_agent_context(self, agent) -> None:
        """Give the selected domain agent the current session history."""
        agent.memory.messages = self._memory.get_chat_history()

    def run(self, input_text: str) -> str:
        """Create a validated plan for every chat request, then execute it."""
        result, _, _ = self.run_with_metrics(input_text)
        return result

    def run_with_metrics(self, input_text: str) -> tuple[str, dict, str]: #用户输入→返回值【ai回复，统计信息，agent名字】
        """Plan every request before invoking the selected domain actions."""
        start = time.time()
        self._memory.add_user_message(input_text)
        try:
            plan = self._create_task_plan(input_text)
            log_event(_LOGGER, "plan_created", actions=plan.actions,
                      candidate_count=plan.candidate_count,
                      exclusions=plan.exclusions)
            result = self._execute_task_plan(plan, input_text)
        except Exception as exc:
            plan = None
            result = _failure_json(exc, "task_planner")

        self._memory.add_ai_message(result)
        elapsed = (time.time() - start) * 1000
        return result, {
            "elapsed_ms": round(elapsed, 1),
            "planner_llm_calls": 1 if plan is not None else 0,
            "llm_calls": 1 if plan is not None else 0,
            "pipeline": ",".join(plan.actions) if plan else "planning_failed",
        }, "task_planner"

    def run_pipeline(self, task_or_agent: TaskType | str, input_text: str) -> str:
        """Run a specific agent pipeline directly (bypass classification).指定Agent运行。"""
        if isinstance(task_or_agent, TaskType):
            agent_name = self._task_to_agent.get(task_or_agent, "data_agent")
            if agent_name is None:
                return self.run(input_text)
        else:
            agent_name = task_or_agent

        agent = self._agents.get(agent_name)
        if agent is None:
            return f"未找到 Agent: {agent_name}"
        try:
            self._sync_agent_context(agent)
            return agent.run(input_text)
        except Exception as e:
            return _failure_json(e, agent_name)

    def explore_dataset(self) -> str:  #只是封装。
        return self._agents["data_agent"].explore_dataset()

    def search_semantic(self, query: str, k: int = 5) -> list[dict]:
        return self._agents["data_agent"].search_semantic(query, k)

    @staticmethod
    def _search_candidate_pool(data_agent, query: str, k: int) -> list[dict]:
        """Merge structured all-table hits with semantic candidate recall.

        Structured hits are placed first so conditions that only exist in an
        associated table (for example language or employment) are not lost
        before the screening stage. The fallback keeps compatibility with
        lightweight test doubles and older DataAgent implementations.
        """
        merged: list[dict] = []
        seen: set[str] = set()
        structured_search = getattr(data_agent, "search_candidates_all_tables", None)
        should_search = getattr(data_agent, "should_use_structured_search", None)
        use_structured = should_search(query) if should_search is not None else True
        if structured_search is not None and use_structured:
            try:
                for item in structured_search(query, k=k):
                    person_id = str(item.get("人员ID", "")).strip()
                    if person_id and person_id not in seen:
                        merged.append(item)
                        seen.add(person_id)
            except Exception:
                # Semantic recall remains available if a secondary table is
                # unavailable or has malformed data.
                pass

        semantic_search = getattr(data_agent, "search_semantic")
        for item in semantic_search(query, k=k):
            person_id = str(item.get("人员ID", "")).strip()
            if person_id and person_id not in seen:
                merged.append(item)
                seen.add(person_id)
            if len(merged) >= k:
                break
        return merged[:k]

    def _create_task_plan(self, input_text: str) -> HiringTaskPlan:
        """Use forced structured output so planning is validated before execution."""
        return structured_completion(
            messages=[{"role": "user", "content": input_text}],
            output_model=HiringTaskPlan,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

    @staticmethod
    def _requirements_text(plan: HiringTaskPlan) -> str:
        lines = [*plan.requirements]
        lines.extend(f"排除条件：{item}" for item in plan.exclusions)
        return "\n".join(lines) or "请根据原始请求核验候选人条件。"

    @staticmethod
    def _format_search_results(results: list[dict], requested_count: int) -> str:
        if not results:
            return "未找到匹配的候选人。"
        lines = [f"找到 {min(len(results), requested_count)} 名候选人：\n"]
        for candidate in results[:requested_count]:
            lines.append(f"  {candidate.get('姓名', '')} ({candidate.get('人员ID', '')})")
            lines.append(f"  {candidate.get('摘要', '')[:120]}\n")
        return "\n".join(lines)

    def _execute_task_plan(self, plan: HiringTaskPlan, input_text: str) -> str:
        """Execute only the actions named in a validated task plan."""
        if plan.clarification_needed:
            return plan.clarification_question or "请补充岗位要求或候选人范围。"

        actions = set(plan.actions)
        if "search" in actions and actions & {"screen", "analyze", "compare", "report", "interview"}:
            return self._run_hiring_pipeline(input_text, plan)

        if actions == {"search"}:
            results = self._search_candidate_pool(
                self._agents["data_agent"], input_text, plan.candidate_count
            )
            return self._format_search_results(results, plan.candidate_count)

        if "screen" in actions and plan.candidate_ids:
            requirements = self._requirements_text(plan)
            outputs = []
            for person_id in plan.candidate_ids:
                raw = self._agents["screening_agent"].screen_candidate(
                    person_id, requirements=requirements
                )
                outputs.append(self._screening_output(raw))
            return json.dumps(outputs, ensure_ascii=False, indent=2)

        if "analyze" in actions and plan.candidate_ids:
            reports = [
                self._agents["analysis_agent"].analyze(person_id, context=self._requirements_text(plan))
                for person_id in plan.candidate_ids
            ]
            return "\n\n".join(reports)

        if "interview" in actions and plan.candidate_ids:
            questions = []
            for person_id in plan.candidate_ids:
                questions.append(
                    self._agents["interview_agent"].generate_questions(
                        person_id, position_info=self._requirements_text(plan)
                    )
                )
            return json.dumps(
                [[question.model_dump(mode="json") for question in group] for group in questions],
                ensure_ascii=False,
                indent=2,
            )

        if actions & {"compare", "report"}:
            self._sync_agent_context(self._agents["report_agent"])
            return self._agents["report_agent"].run(input_text)

        self._sync_agent_context(self._agents["data_agent"])
        return self._agents["data_agent"].run(input_text)

    @staticmethod
    def _screening_output(raw):
        """Normalize ScreeningAgent's programmatic result to a JSON-safe dict."""
        from agents.screening_agent import ScreeningOutput

        if isinstance(raw, ScreeningOutput):
            return raw.model_dump(mode="json")
        if isinstance(raw, str):
            return ScreeningOutput.model_validate_json(raw).model_dump(mode="json")
        return ScreeningOutput.model_validate(raw).model_dump(mode="json")

    def _run_hiring_pipeline(self, input_text: str, plan: HiringTaskPlan) -> str:
        """Run Data -> Screening -> Analysis -> Report for compound hiring requests.

        Each stage receives concrete IDs and evidence from the previous stage.
        This deliberately avoids treating a vector-search summary as a hiring
        decision.
        """
        requested_count = plan.candidate_count
        candidate_pool_size = min(max(requested_count * 4, 10), 20)
        log_event(_LOGGER, "pipeline_started", requested_count=requested_count,
                  candidate_pool_size=candidate_pool_size)

        try:
            if plan.candidate_ids:
                retrieved = [{"人员ID": person_id, "姓名": "", "摘要": ""}
                             for person_id in plan.candidate_ids]
            else:
                retrieved = self._search_candidate_pool(
                    self._agents["data_agent"], input_text, candidate_pool_size
                )
        except Exception as exc:
            return _failure_json(exc, "data_agent")

        # Chroma IDs are expected to be unique, but retain that invariant at
        # the orchestration boundary before issuing costly screening calls.
        candidates = []
        seen_ids: set[str] = set()
        for candidate in retrieved:
            person_id = str(candidate.get("人员ID", "")).strip()
            if person_id and person_id not in seen_ids:
                candidates.append(candidate)
                seen_ids.add(person_id)

        if not candidates:
            return "未找到可供筛选的候选人。"

        screening_agent = self._agents["screening_agent"]
        requirements = self._requirements_text(plan)
        screened: list[dict] = []
        screening_failures: list[str] = []
        for candidate in candidates:
            person_id = str(candidate["人员ID"])
            try:
                screening = self._screening_output(
                    screening_agent.screen_candidate(person_id, requirements=requirements)
                )
            except Exception:
                screening_failures.append(person_id)
                continue
            screened.append({"candidate": candidate, "screening": screening})

        if not screened:
            return "候选人已召回，但筛选阶段未能产生可审核的结构化结果。"

        # Never let an explicit rejection enter the final ranking.  Reviews
        # remain eligible only when there are not enough confirmed passes and
        # are clearly marked for a human decision in the final evidence.
        passes = [item for item in screened if item["screening"]["决策"] == "pass"]
        reviews = [item for item in screened if item["screening"]["决策"] == "review"]
        ranked = sorted(
            passes + reviews,
            key=lambda item: float(item["screening"]["匹配分数"]),
            reverse=True,
        )
        finalists = ranked[:requested_count]
        if not finalists:
            return "已完成筛选，但没有候选人满足可进入比较的最低条件。"

        analysis_agent = self._agents["analysis_agent"]
        for item in finalists:
            person_id = str(item["candidate"]["人员ID"])
            try:
                item["analysis"] = analysis_agent.analyze(person_id, context=requirements)
            except Exception as exc:
                item["analysis"] = _failure_json(exc, "analysis_agent")

        evidence = [
            {
                "人员ID": item["candidate"]["人员ID"],
                "姓名": item["candidate"].get("姓名", ""),
                "检索摘要": item["candidate"].get("摘要", ""),
                "筛选结果": item["screening"],
                "深度分析": item["analysis"],
            }
            for item in finalists
        ]
        candidate_ids = [str(item["candidate"]["人员ID"]) for item in finalists]
        report_prompt = f"""请基于以下经过筛选和分析的候选人证据，完成最终招聘比较。

岗位名称：{plan.position_name or '未提供'}
岗位要求：{requirements}
候选人ID：{', '.join(candidate_ids)}
证据（只可引用其中已给出的事实；若决策为 review，必须保留人工复核提示）：
{json.dumps(evidence, ensure_ascii=False)}

请输出：
1. 候选人对比表（技能、经验、教育、总分、风险）
2. 从高到低的排序
3. 最适合人选及可核查的推荐理由
4. 未满足或需要人工复核的条件
"""
        try:
            self._sync_agent_context(self._agents["report_agent"])
            report = self._agents["report_agent"].run(report_prompt)
        except Exception as exc:
            report = _failure_json(exc, "report_agent")

        header = (
            f"已从 {len(candidates)} 名召回候选人中完成筛选，"
            f"选出 {len(finalists)} 名进入比较。"
        )
        if screening_failures:
            header += f" {len(screening_failures)} 名候选人筛选失败，未参与排序。"
        log_event(_LOGGER, "pipeline_finished", retrieved_count=len(candidates),
                  screened_count=len(screened), finalist_count=len(finalists))
        return f"{header}\n\n{report}"

