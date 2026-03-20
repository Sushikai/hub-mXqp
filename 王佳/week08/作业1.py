"""
作业1: 参考 04_Pydantic与Tools.py 中的tools的实现，搭建出一个文本翻译智能体，自动识别需要翻译的文本（原始语种、目标语种，待翻译的文本）。
帮我将good！翻译为中文 -》 原始语种、目标语种，待翻译的文本
"""
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-7458206891744b7abu6d6f7366fecdd5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "system",
                "content": "你是一个文本翻译助手，能够自动识别用户输入中的源语言、目标语言和待待翻译文本。"
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        try:
            # 提取的参数
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

    def translate(self, source_language, target_language, text):
        """调用翻译API获取翻译结果"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"你是一名专业翻译，请将{source_language}文本翻译为{target_language}。只输出翻译结果，不要其它内容。"
                },
                {
                    "role": "user",
                    "content": f"请将以下内容从{source_language}翻译为{target_language}：{text}"
                }
            ]
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"翻译时出错：{str(e)}")
            return f"翻译失败：{str(e)}"

class TranslationModel(BaseModel):
    """
    翻译模型
    """
    source_language: str = Field(..., description="原始语种")
    target_language: str = Field(..., description="目标语种")
    text_to_translate: str = Field(..., description="待翻译的文本")

# 提取参数
result = ExtractionAgent('qwen3.5-plus').call(
    "请将good!翻译为中文",
    TranslationModel
)
print(result)

if result:
    print(f"\n=== 翻译结果 ===")
    print(f"原文：{result.text_to_translate}")
    print(f"语种：{result.source_language} -> {result.target_language}")

    translator = ExtractionAgent("qwen3.5-plus")
    translated_text = translator.translate(
        result.source_language,
        result.target_language,
        result.text_to_translate
    )
    print(f"译文：{translated_text}")
