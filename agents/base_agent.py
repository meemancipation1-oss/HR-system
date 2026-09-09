"""Base agent class — common interface for all HR agents.

Pure tool-calling loop with no framework dependencies.
Features:
- Timing & token tracking for performance evaluation
- Parallel tool execution via ThreadPoolExecutor
"""

#所有 Agent 共用的底层执行引擎,ReAct Agent Loop

from __future__ import annotations
from abc import ABC, abstractmethod
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from core.llm import chat_completion, extract_content, extract_tool_calls, extract_message, structured_completion
from core.memory import AgentMemory
from core.tools import get_all_tools_schema, execute_tool, ToolDef
from config.settings import TOOL_TIMEOUT_SECONDS, MAX_RETRIES
from core.observability import configure_logging, log_event, current_trace_context, trace_context
from core.runtime import RuntimeFailure

DEFAULT_MAX_ITERATIONS = 10
_PARALLEL_WORKERS = 4
_LOGGER = configure_logging(__import__("config.settings", fromlist=["LOG_PATH"]).LOG_PATH)


class AgentMetrics:
    """Performance metrics collected during a single agent run."""
#记录每次 Agent 运行的性能数据

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.llm_calls: int = 0
        self.tool_calls: int = 0
        self.parallel_batches: int = 0

    @property  #@property 的作用就是：把一个方法变成"属性"来访问。把一个函数变成一个属性，会直接得到结果
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def summary(self) -> str: #拼接
        return (
            f"⏱ {self.elapsed_ms:.0f}ms | " #保留0位小数
            f"LLM {self.llm_calls}次 | "
            f"工具 {self.tool_calls}次 | "
            f"并行 {self.parallel_batches}批"
        )

    def dict(self) -> dict:
        return {
            "elapsed_ms": round(self.elapsed_ms, 1),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "parallel_batches": self.parallel_batches,
        }
#把一个 Python 对象转换成一个普通的字典（dict），方便打印、保存、转换成 JSON 或返回给前端。

class BaseAgent(ABC):  #Abstract Base Class（抽象基类），定义了所有 Agent 的通用工作流程，必须：XX(BaseAgent)继承
    """Abstract base for all HR agents."""

    name: str = "base"
    system_prompt: str = "You are a helpful HR assistant."
    tools: list[ToolDef] = []

    def __init__(self, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> None:
        self.memory = AgentMemory()  #创建聊天记忆
        self.max_iterations = max_iterations #防止AI死循环
        self._executor = ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) #创建线程池

    @abstractmethod #抽象方法。BaseAgent：不实现。必须：子类：实现。
    def run(self, input_text: str) -> str:
        ...


#最重要的部分，loop
    def run_with_metrics(self, input_text: str) -> tuple[str, AgentMetrics]:
        """Execute and return (output, metrics)."""
        metrics = AgentMetrics() #统计：
        metrics.start_time = time.time() #记录开始时间

        prompt = self.system_prompt #系统 Prompt
        tools_schema = get_all_tools_schema() #得到所有tool

        messages: list[dict] = [{"role": "system", "content": prompt}]  #system加入历史再加入user
        messages.extend(self.memory.get_chat_history())
        messages.append({"role": "user", "content": input_text})

        for iteration in range(self.max_iterations):  #Tool Loop
            log_event(_LOGGER, "agent_iteration", agent=self.name, iteration=iteration + 1)
            response = chat_completion(
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
            )#调用大模型
            metrics.llm_calls += 1 #统计次数

            assistant_msg = extract_message(response) #保存Assistant消息
            messages.append(assistant_msg)

            tool_calls = extract_tool_calls(response)  #判断是否有tool

            if not tool_calls: #没有tool：直接：结束。保存：Memory。返回：结果。
                output = extract_content(response)
                metrics.end_time = time.time()
                self.memory.add_user_message(input_text)
                self.memory.add_ai_message(output)
                return output, metrics

            # Parallel tool execution
            metrics.tool_calls += len(tool_calls)  #并行tool
            if len(tool_calls) > 1:
                metrics.parallel_batches += 1

            futures = {} #未来会有结果，结果的占位符
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                future = self._executor.submit(
                    self._safe_execute_tool, fn_name, fn_args, current_trace_context()
                )
                #submit 的意思就是：把任务交给线程池。
                futures[future] = tc #Future 不知道自己属于哪个 Tool。建立映射

            try:
                completed = as_completed(futures, timeout=TOOL_TIMEOUT_SECONDS)
                for future in completed: #按照完成顺序优先提交
                    tc = futures[future] #根据future找到对应tool
                    try:
                        result = future.result()
                    except Exception as e:
                        result = f"工具执行出错: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    })
            except FuturesTimeoutError:
                for future, tc in futures.items():
                    if not future.done():
                        future.cancel()
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"工具执行超时（>{TOOL_TIMEOUT_SECONDS:g}s）",
                        })

        metrics.end_time = time.time()
        fallback = "处理超时，请简化您的请求。"
        log_event(_LOGGER, "agent_incomplete", agent=self.name, reason="max_iterations",
                  max_iterations=self.max_iterations)
        failure = RuntimeFailure(
            status="incomplete", error_code="max_iterations",
            message=fallback, retryable=True,
            details={"max_iterations": self.max_iterations},
        ).model_dump_json(ensure_ascii=False)
        self.memory.add_user_message(input_text)
        self.memory.add_ai_message(failure)
        return failure, metrics

    def _safe_execute_tool(self, name: str, args: str, context: dict | None = None) -> str: #保证不会因为tool报错而直接结束进程
        with trace_context(**(context or {})):
            started = time.time()
            last_error: Exception | None = None
            log_event(_LOGGER, "tool_started", agent=self.name, tool=name)
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = execute_tool(name, args)
                    log_event(_LOGGER, "tool_finished", agent=self.name, tool=name,
                              latency_ms=round((time.time() - started) * 1000, 1),
                              retry_count=attempt, success=True)
                    return result
                except (ValueError, TypeError, KeyError) as exc:
                    last_error = exc
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES:
                        continue
            log_event(_LOGGER, "tool_failed", agent=self.name, tool=name,
                      latency_ms=round((time.time() - started) * 1000, 1),
                      retry_count=MAX_RETRIES, success=False,
                      error_type=type(last_error).__name__ if last_error else "UnknownError")
            return f"工具 [{name}] 执行出错: {last_error}"

    def _execute(self, input_text: str, system_prompt: str | None = None) -> str: #包装，方便调用
        output, _ = self.run_with_metrics(input_text)
        return output

    def _execute_structured(
        self,
        input_text: str,
        output_model: type,
        system_prompt: str | None = None,
    ) -> str:
        """Run tool-calling loop to gather context, then force structured output.

        Pattern:
          1. First pass: full tool loop to gather data (calls search_individual, etc.)
          2. Second pass: forced tool calling to produce structured output
        """
        prompt = system_prompt or self.system_prompt
        output_attempted = False

        try:
            # Step 1: gather data using real tools + structured output together
            # We pass both data tools AND the output_result tool so the model can
            # gather context and then structure the result.
            all_tools = list(get_all_tools_schema())
            output_tool = {
                "type": "function",
                "function": {
                    "name": "output_result",
                    "description": f"输出结构化结果，遵循 {output_model.__name__} schema",
                    "parameters": output_model.model_json_schema(),
                },
            }
            all_tools.append(output_tool)

            messages = [{"role": "system", "content": prompt}]
            messages.extend(self.memory.get_chat_history())
            messages.append({"role": "user", "content": input_text})

            for iteration in range(self.max_iterations):
                log_event(_LOGGER, "agent_iteration", agent=self.name,
                          iteration=iteration + 1, structured_output=True)
                response = chat_completion(
                    messages=messages,
                    tools=all_tools,
                    tool_choice="auto",
                )

                assistant_msg = extract_message(response)
                messages.append(assistant_msg)
                tool_calls = extract_tool_calls(response)

                if not tool_calls:
                    output = extract_content(response)
                    self.memory.add_user_message(input_text)
                    self.memory.add_ai_message(output)
                    return output

                # Check if the model chose to call output_result
                for tc in tool_calls:
                    if tc["function"]["name"] == "output_result":
                        args = json.loads(tc["function"]["arguments"])
                        result = output_model(**args)
                        output = result.model_dump_json(ensure_ascii=False, indent=2)
                        self.memory.add_user_message(input_text)
                        self.memory.add_ai_message(output)
                        return output

                # Execute real tools with the same batch timeout as the normal loop.
                # Do not use the executor as a context manager: its implicit
                # shutdown(wait=True) would defeat the timeout for already-running tools.
                pool = ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS)
                fut_to_tc = {}
                try:
                    for tc in tool_calls:
                        fut = pool.submit(
                            self._safe_execute_tool, tc["function"]["name"],
                            tc["function"]["arguments"], current_trace_context()
                        )
                        fut_to_tc[fut] = tc
                    try:
                        for fut in as_completed(fut_to_tc, timeout=TOOL_TIMEOUT_SECONDS):
                            tc = fut_to_tc[fut]
                            try:
                                result = fut.result()
                            except Exception as e:
                                result = f"出错: {e}"
                            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
                    except FuturesTimeoutError:
                        for fut, tc in fut_to_tc.items():
                            if not fut.done():
                                fut.cancel()
                                log_event(_LOGGER, "tool_timeout", agent=self.name,
                                          tool=tc["function"]["name"],
                                          timeout_seconds=TOOL_TIMEOUT_SECONDS)
                                messages.append({
                                    "role": "tool", "tool_call_id": tc["id"],
                                    "content": f"工具执行超时（>{TOOL_TIMEOUT_SECONDS:g}s）",
                                })
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)

            return RuntimeFailure(
                status="incomplete", error_code="max_iterations",
                message="Agent 达到最大执行轮次，未能完成结构化输出。",
                retryable=True, details={"max_iterations": self.max_iterations},
            ).model_dump_json(ensure_ascii=False)

        except Exception as e:
            log_event(_LOGGER, "agent_failed", agent=self.name,
                      error_type=type(e).__name__)
            is_timeout = "timeout" in type(e).__name__.lower()
            return RuntimeFailure(
                status="incomplete" if is_timeout else "error",
                error_code="llm_timeout" if is_timeout else "invalid_output",
                message="模型调用超时，请稍后重试。" if is_timeout else "模型输出无法通过结构化校验。",
                retryable=True, details={"error_type": type(e).__name__},
            ).model_dump_json(ensure_ascii=False)


__all__ = ["BaseAgent", "AgentMetrics"]

#BaseAgent 是整个 Agent 框架的抽象基类，封装了所有 Agent 共用的执行流程，
# 包括 Prompt 组织、Memory 管理、Tool Calling、线程池并发执行、Agent 循环推理以及性能统计。
# 核心方法 run_with_metrics() 实现了典型的 ReAct（Reason + Act）模式：先调用 LLM 判断是否需要使用工具；
# 如果需要，则并行执行多个 Tool，并将结果作为 tool 消息追加到上下文，再次交给 LLM 推理；
# 当模型不再返回 Tool Call 时，输出最终答案并更新 Memory。对于需要结构化结果的场景，_execute_structured() 在普通 Tool 的基础上额外注册 output_result 工具，
# 使模型能够在完成数据检索后，按照 Pydantic Schema 输出符合要求的结构化数据。这样整个框架既支持多轮工具调用，也支持可靠的结构化输出。
