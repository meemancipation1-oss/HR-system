import json
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

MODEL = "deepseek-chat"

client = AsyncOpenAI(
    api_key="",
    base_url=""
)

class SearchJobParams(BaseModel):
    keyword: str = Field(description="search a phd job")

def search_job(keyword: str):
    return f"zhaodao{keyword}job"

TOOLS ={
    "search_job": search_job
}

TOOLS_SCHEMA=[
    {
        "type":"function",
        "function":{
            "name":"search_job",
            "description":"search a phd job",
            "parameters": SearchJobParams.model_json_schema()
        }
    }
]

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request:ChatRequest):
    messages=[
        {
            "role":"user",
            "content":request.message
        }
    ]
    while True:
        response = await client.chat.completions.create(
            model="deepseek",
            messages=messages,
            tool_choice="auto",
            tools=TOOLS_SCHEMA
        )

        assistant = response.choices[0].message
        messages.append(assistant.model_dump())

        if not assistant.tool_calls:
            break

        for tool_call in assistant.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            try:
                reslut = TOOLS[name](**args)

            except Exception as e:
                reslut=f"TOOL ERROR:{str(e)}"

            messages.append(
                {
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content": reslut
                }
            )

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
        media_type="text/event-stream "
    )

