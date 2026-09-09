import json

from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse
import uvicorn
# ====================================================
# FastAPI
# ====================================================

app = FastAPI()


# ====================================================
# OpenAI Client
# ====================================================

client = AsyncOpenAI(
    api_key="你的API Key",
    base_url="你的Base URL"
)


# ====================================================
# Tool
# ====================================================
class SearchJobParams(BaseModel):
    keyword: str = Field(description="岗位关键词")

def search_job(keyword: str):
    """
    模拟岗位搜索
    """

    print(f"正在搜索：{keyword}")

    return f"找到岗位：{keyword} 算法工程师"


# 所有工具统一放这里
TOOLS = {
    "search_job": search_job
}


# ====================================================
# Request / Response
# ====================================================

class ChatRequest(BaseModel):
    message: str




# ====================================================
# Chat API
# ====================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    # 保存聊天记录
    messages = [
        {
            "role": "user",
            "content": request.message
        }
    ]

    while True:

        # -----------------------------
        # 第一次（或者后续）请求 GPT 不开流式
        # -----------------------------

        response = await client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_job",
                        "description": "搜索岗位",

                        "parameters": SearchJobParams.model_json_schema(),
                    }
                }
            ],

            tool_choice="auto"
        )

        assistant = response.choices[0].message

        # 保存 Assistant 回复
        messages.append(assistant.model_dump())

        # -------------------------------------------------
        # 没有 Tool Call
        # GPT 已经回答完了
        # -------------------------------------------------

        if assistant.tool_calls:

            for tool_call in assistant.tool_calls:
                name = tool_call.function.name

                args = json.loads(
                    tool_call.function.arguments
                )

                result = TOOLS[name](**args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            # -----------------------------
            # 第二次请求（开启流）
            # -----------------------------

        async def generate():  # 异步，负责接收每个 Chunk，提取 delta.content，并用 yield 一块一块地产生数据。
            stream = await client.chat.completions.create(
                model="deepseek-v4",
                messages=messages,
                stream=True
            )

            async for chunk in stream:  # 一边等待，一边遍历。

                if not chunk.choices:  # 没有的话跳过继续下一轮
                    continue

                delta = chunk.choices[0].delta  # 流式没有message，只有增量

                if delta.content:  # 判断是否是文字，有可能是toolcall
                    yield delta.content  # 区别于return，不会直接结束进程，暂停等下一次

        return StreamingResponse(  # 负责把 generate() 产生的数据立即发送给前端
            generate(),  # 把生成器交给内部
            media_type="text/event-stream"
        )


if __name__ == "__main__":  #启动入口
        import uvicorn

        uvicorn.run(
            "all agent:app",  # main.py 中的 app
            host="0.0.0.0",  # 监听所有IP
            port=8000,  # 端口
            reload=True  # 修改代码自动重启（开发环境）
        )