from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

# ==========================
# 1. 创建 FastAPI 应用
# ==========================
app = FastAPI()

# ==========================
# 2. 创建 OpenAI Client
# ==========================
client = AsyncOpenAI(
    api_key="你的API Key",
    base_url="你的Base URL"
)

# ==========================
# 3. 定义请求数据格式
# ==========================
class ChatRequest(BaseModel):
    message: str

# ==========================
# 4. 定义返回数据格式
# ==========================
class ChatResponse(BaseModel):
    answer: str

# ==========================
# 5. 定义聊天接口
# ==========================
#定义聊天接口，当有人发送 POST 请求到 /chat 时，就执行下面这个函数。/chat它就是网址的一部分。可以随意命名
#网址绑定函数
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    response = await client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": request.message
            }
        ]
    )

    answer = response.choices[0].message.content

    return ChatResponse(
        answer=answer
    )