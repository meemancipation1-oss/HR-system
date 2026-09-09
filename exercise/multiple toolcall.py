import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

app = FastAPI() #创建 FastAPI 应用

# ========================================
# LLM
# ========================================

MODEL = "deepseek-chat" #定义一个变量

#创建一个客户端,与LLM建立链接
client = AsyncOpenAI(
    api_key="你的API Key",
    base_url="你的Base URL"
)

# ========================================
# Tool
# ========================================

#定义 Tool 参数，必须叫keyword，类型：str,field这是参数说明
class SearchJobParams(BaseModel):
    keyword: str = Field(description="岗位关键词")


#Tool 函数,可以有很多功能
def search_job(keyword: str):
    print(f"搜索岗位：{keyword}")
    return f"找到岗位：{keyword}算法工程师"

#建立函数名字到python函数的映射
TOOLS = {
    "search_job": search_job
}

#介绍tool
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_job",
            "description": "搜索岗位",
            "parameters": SearchJobParams.model_json_schema()
        }
    }
]

# ========================================
# Request
# ========================================

class ChatRequest(BaseModel):
    message: str

# ========================================
# API
# ========================================

@app.post("/chat") #注册一个 POST 接口
async def chat(request: ChatRequest):

    messages = [
        {
            "role": "user",
            "content": request.message
        }
    ]

    # -----------------------------
    # Tool Calling 循环
    # -----------------------------
    while True:

        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )

        assistant = response.choices[0].message #取第一条回复。可能有content也可能是toolcall

        messages.append(assistant.model_dump())

        # 没有 Tool，结束循环
        if not assistant.tool_calls:
            break

        # 执行所有 Tool
        for tool_call in assistant.tool_calls:

            tool_name = tool_call.function.name

            args = json.loads(
                tool_call.function.arguments
            )

            try:
                result = TOOLS[tool_name](**args)

            except Exception as e:
                result = f"Tool Error: {str(e)}"  #防止 Tool 崩掉

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            )

    # -----------------------------
    # 最后一轮 Streaming
    # -----------------------------
    async def generate():

        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )

        async for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                yield delta.content

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )