from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-b8d6efe8169abu51a91a13b8fe8fd99a", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 本质是function call
# 可以传入多个待选函数，让大模型选择其中一个
# 传的是我们的函数的描述，让大模型选择，生成调用这个函数的传入参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "Translate",
            "description": "根据用户提供的信息翻译成对应的语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_language": {
                        "description": "原始语种",
                        "title": "Current_language",
                        "type": "string",
                    },
                    "target_language": {
                        "description": "目标语种",
                        "title": "Target_language",
                        "type": "string",
                    },
                    "text": {
                        "description": "待翻译的文本",
                        "title": "Text",
                        "type": "string",
                    },
                },
                "required": ["current_language", "target_language", "text"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "帮我将好！翻译为英文"
    }
]

# 大模型选择了一个函数，生成了函数的调用过程， 这也是agent 的核心功能
response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools, # 生成函数的调用方式，并不是所有的模型都支持（某些比较小的模型不支持）
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls[0].function)

