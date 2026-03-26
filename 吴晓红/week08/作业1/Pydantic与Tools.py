from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-xxx", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "transfer",
            "description": "根据提供的信息翻译文本",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "description": "原始语种",
                        "title": "source",
                        "type": "string",
                    },
                    "target": {
                        "description": "目标语种",
                        "title": "target",
                        "type": "string",
                    },
                    "content": {
                        "description": "待翻译的文本",
                        "title": "content",
                        "type": "string",
                    },
                },
                "required": ["date", "departure", "destination"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "帮我将good！翻译为中文"
    }
]

response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools, # 生成函数的调用方式，并不是所有的模型都支持（某些比较小的模型不支持）
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls[0].function)
