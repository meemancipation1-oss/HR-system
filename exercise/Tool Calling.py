import json
from openai import OpenAI
def search_job(keywords:str): #这里定义了一个普通 Python 函数。 参数：类型
    '''模拟搜索岗位'''
    return f'找到岗位：{keywords}算法工程师'

TOOLS = {'search_job': search_job} #python字典，左边是字符串，右边是函数对象，建立"工具名称"到"Python函数"的映射。

#创建SDK客户端
client = OpenAI(
    api_key='',
    base_url=''
)

messages = [
    {
        'role':'user',
        'content':'帮我找到博士岗位'
    }
]

while True:
    response = client.chat.completions.create(
        model = 'gpt-5',
        messages = messages,
        #告诉模型有哪些工具可以用
        tools = [
            {
                'type': 'function', #类型是函数工具
                'function' : {
                    "name": "search_job", #工具名字。
                    "description": "搜索岗位", #工具描述
                    "parameters": {  #函数参数说明（JSON Schema），给gpt看
                        "type": "object",  #参数整体为一个json对象
                        "properties": {   #表示有哪些字段
                            "keyword": {   #表示参数是keyword，类型是string
                                "type": "string"
                            }
                        },
                        "required":[ "keyword"] #说明必须有kerword
                    }
                }
            }
        ],
        tool_choice = 'auto'
    )

#Tool 的结构正确写法应该是：
#{
#    "type": "function",
#    "function": {
#        ...
#   }
#}
    assistant = response.choices[0].message #获取回复
    messages.append(assistant.model_dump()) #添加回复到全文

    if not assistant.tool_calls:
        print(assistant.content)
        break

    for tool_call in assistant.tool_calls:
        name = tool_call.function.name #得到工具名字
        args = json.loads(tool_call.function.arguments) #它的作用是把模型返回的 JSON 字符串解析成 Python 字典。
        result = TOOLS[name](**args) #**args 是 Tool Calling 中把模型生成的参数传递给 Python 函数最常见、最方便的写法。
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content":result,
            }
        )  #把工具执行结果作为一条新的消息发回给模型。