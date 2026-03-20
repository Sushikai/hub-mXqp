from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0abuabu58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "Translation",
            "description": "提取翻译任务中的关键信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_language": {
                        "type": "string",
                        "description": "原始语种"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语种"
                    },
                    "text": {
                        "type": "string",
                        "description": "待翻译文本"
                    }
                },
                "required": ["source_language", "target_language", "text"]
            }
        }
    }
]

messages = [
    {
        "role": "user",
        "content": "帮我将 good! 翻译为中文"
    }
]

response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print("===== 原始 tool 调用 =====")
print(response.choices[0].message.tool_calls[0].function)

# 封装 Agent
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

# 定义翻译任务
class Translation(BaseModel):
    """文本翻译信息解析"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    text: str = Field(description="待翻译文本")

# 测试
agent = ExtractionAgent(model_name="qwen-plus")

result = agent.call("帮我将 good! 翻译为中文", Translation)
print("===== Agent解析结果 =====")
print(result)

# 扩展测试
examples = [
    "把 hello world 翻译成中文",
    "请将 你好 翻译为英文",
    "Translate bonjour to Chinese",
]

for text in examples:
    res = agent.call(text, Translation)
    print("\n输入:", text)
    print("输出:", res)
