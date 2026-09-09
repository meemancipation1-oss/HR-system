"""LLM interface — lightweight wrapper around OpenAI-compatible API and embeddings.

Uses the native `openai` SDK (no LangChain). Supports:
  - DeepSeek / OpenAI / any OpenAI-compatible API for chat
  - sentence-transformers (local) for embeddings
  - Streaming for real-time responses
  - Structured output via forced tool calling

Structured output via forced tool calling (since DeepSeek doesn't support
response_format=json_object natively).
"""

from __future__ import annotations
import json
from typing import Any, Generator
from exercise.pydanticEX import BaseModel
from openai import OpenAI

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LLM_MODEL,
    TEMPERATURE,
    MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
)
from core.observability import configure_logging, log_event
from config.settings import LOG_PATH
# 从配置文件读取 API Key、模型名、温度等参数

_client: OpenAI | None = None
_logger = configure_logging(LOG_PATH)
# 全局变量，缓存 OpenAI 客户端实例，避免每次调用都重新创建


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            max_retries=MAX_RETRIES,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    return _client
# 惰性初始化：第一次调用时才创建客户端，之后复用
# 注意：base_url 默认是 https://api.deepseek.com → 支持 DeepSeek

#对 OpenAI（或 DeepSeek/OpenAI兼容接口）SDK 的一层封装（Wrapper），目的：屏蔽 SDK 的细节，让项目其他地方统一调用 chat_completion() 即可。
def chat_completion( #定义一个封装函数，以后别人不用写：client.chat.completions.create(...)，只需要chat_completion(messages)
    messages: list[dict], #这是聊天历史。里面每个元素都是字典。
    model: str | None = None,
    temperature: float | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    stream: bool = False, #不开启流式。
    **kwargs: Any, #接收其它所有参数。
) -> dict | Generator[dict, None, None]: #普通模式返回dict，流式模式返回generator就是生成器
    """Call the chat completion API.

    If stream=True, yields response chunks as dicts (for SSE).
    Otherwise returns the full response dict.
    """
    client = get_client() #全局单例复用
    kwargs.setdefault("model", model or LLM_MODEL)
    kwargs.setdefault("temperature", temperature or TEMPERATURE)
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    if stream:
        return _stream_completion(client, messages, **kwargs) #一个函数专门处理stream

    started = __import__("time").time()
    log_event(_logger, "llm_started", model=kwargs.get("model"))
    try:
        resp = client.chat.completions.create(messages=messages, **kwargs)
        result = resp.model_dump()
        usage = result.get("usage") or {}
        log_event(_logger, "llm_finished", model=kwargs.get("model"), success=True,
                  latency_ms=round((__import__("time").time() - started) * 1000, 1),
                  prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
        return result
    except Exception as exc:
        log_event(_logger, "llm_failed", model=kwargs.get("model"), success=False,
                  latency_ms=round((__import__("time").time() - started) * 1000, 1),
                  error_type=type(exc).__name__)
        raise

#为什么要封装这一个函数：这是一个统一的 API 封装层（Wrapper）。它负责获取全局单例 Client、设置默认模型和温度参数、按需注入 Tool 配置，并根据 stream 参数分别处理普通响应和流式响应。业务层只调用 chat_completion()，无需关心底层 SDK 的实现细节。如果未来切换模型（例如 GPT、DeepSeek、Qwen）或修改默认参数，只需要改这一层即可，不需要修改所有业务代码，这样可维护性和扩展性都更好。这也是实际项目中常见的分层设计思想：业务层 → 封装层 → SDK → API 服务。

def _stream_completion(client: OpenAI, messages: list[dict], **kwargs) -> Generator[dict, None, None]:
    """Stream chunks from the API, yielding dicts with 'content' or 'tool_calls'."""
    stream = client.chat.completions.create(messages=messages, stream=True, **kwargs) #开启流式请求
    collected: dict[str, Any] = {"content": "", "tool_calls": []} #这是一个缓存。

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta #delta为增量，每一次stream增加
        if not delta:
            continue

        # Text content
        if delta.content:
            collected["content"] += delta.content  #拼接
            yield {"type": "content", "delta": delta.content, "full": collected["content"]}
        #不是return，return结束而stream不能结束

        # Tool calls (partial)
        if delta.tool_calls:  #调用tool必须拼接好了之后再分析
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                while len(collected["tool_calls"]) <= idx:
                    collected["tool_calls"].append({"id": "", "function": {"name": "", "arguments": ""}})
                if tc_delta.id:
                    collected["tool_calls"][idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        collected["tool_calls"][idx]["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        collected["tool_calls"][idx]["function"]["arguments"] += tc_delta.function.arguments

    # Final message with tool_calls 查找有没有tool call
    if any(tc["id"] for tc in collected["tool_calls"]):
        yield {"type": "tool_calls", "tool_calls": collected["tool_calls"]}
    else:
        yield {"type": "done", "content": collected["content"]}

#这段代码实现了流式响应的统一处理。它首先开启 stream=True 的 Chat Completion 请求，然后通过 for chunk in stream 持续读取服务器返回的增量数据（delta）。对于普通文本，会将 delta.content 累积到 collected["content"] 中，并使用 yield 实时返回给前端，实现打字机效果。对于 tool_calls，由于函数名和 arguments 可能会被拆分到多个 chunk 中，因此根据 index 定位对应的 Tool，并持续拼接 arguments 字符串。流结束后，如果存在 Tool Call，就返回完整的 tool_calls 供 Agent 执行；否则返回 done，表示文本输出完成。
def extract_content(response: dict) -> str:
    #从 API 响应中提取 LLM 生成的文本内容
    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content", "") or ""
# API 返回结构: {"choices": [{"message": {"content": "...", "tool_calls": [...]}}]}

def extract_tool_calls(response: dict) -> list[dict]:
    #从 API 响应中提取 tool_calls 指令
    choices = response.get("choices", [])
    if not choices:
        return []
    msg = choices[0].get("message", {})
    return msg.get("tool_calls", []) or []
# 如果没有 tool_calls，返回空列表，表示 LLM 想直接回复文本

def extract_message(response: dict) -> dict:
    #提取完整的 assistant 消息（包含 content + tool_calls）
    choices = response.get("choices", [])
    if not choices:
        return {"role": "assistant", "content": ""}
    return choices[0].get("message", {"role": "assistant", "content": ""})
## 这里返回的是整个 message dict，直接可以追加到 messages 列表

# ── Structured Output ──
#不让 LLM 返回一段自由文本，而是强制它返回符合指定 Pydantic 模型的数据。
def structured_completion(
    messages: list[dict],
    output_model: type[BaseModel],
    model: str | None = None,
    temperature: float | None = None,
    system_prompt: str | None = None,
) -> BaseModel:
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
#就是把 System Prompt 插到最前面

    tools = [
        {
            "type": "function",
            "function": {
                "name": "output_result",
                "description": f"输出结构化结果，遵循 {output_model.__name__} schema",
                "parameters": output_model.model_json_schema(),
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "output_result"}}
#强制格式化输出

    response = chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
    )

    calls = extract_tool_calls(response) #提取tool call
    if calls:
        args = json.loads(calls[0]["function"]["arguments"]) #解析 JSON
        return output_model(**args) #生成 Pydantic 对象

    content = extract_content(response)
    if content:
        try:
            data = json.loads(content)
            return output_model(**data)
        except (json.JSONDecodeError, Exception):
            pass

    raise ValueError(f"无法从响应中解析 {output_model.__name__}:\n{content}")

##这段代码实现了基于 Tool Calling 的结构化输出。首先根据传入的 Pydantic BaseModel 自动生成 JSON Schema（model_json_schema()），并将其注册为一个名为 output_result 的虚拟函数。同时通过 tool_choice 强制模型调用该函数，从而保证模型返回的数据符合预定义的 Schema。收到响应后，优先解析 Tool Call 中的 arguments，将 JSON 字符串转换为字典，再实例化为对应的 Pydantic 对象，实现自动的数据校验和类型转换。如果模型不支持 Tool Calling，则退化为解析普通文本中的 JSON 内容。整个设计相比 Prompt 要求输出 JSON 更稳定、更适合生产环境。

# ── Embeddings ──

from sentence_transformers import SentenceTransformer

_embedding_model: SentenceTransformer | None = None


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name, device="cpu")
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embeddings()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


__all__ = [
    "get_client",
    "chat_completion",
    "extract_content",
    "extract_tool_calls",
    "extract_message",
    "structured_completion",
    "get_embeddings",
    "embed_texts",
    "embed_query",
]
