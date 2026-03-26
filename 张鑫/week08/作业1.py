import json
import os
import sys

from openai import OpenAI
from pydantic import BaseModel, Field


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def build_client() -> OpenAI:
    api_key = os.getenv('FE8_API_KEY')
    if not api_key:
        raise ValueError('Missing FE8_API_KEY.')

    base_url = os.getenv('FE8_BASE_URL')
    if not base_url:
        raise ValueError('Missing FE8_BASE_URL.')

    return OpenAI(api_key=api_key, base_url=base_url)


client = build_client()
MODEL_NAME = os.getenv('FE8_MODEL', 'qwen-plus')


class TranslationTask(BaseModel):
    source_language: str = Field(description='原始语种，例如英文、中文、日文')
    target_language: str = Field(description='目标语种，例如中文、英文、法文')
    text: str = Field(description='待翻译的文本')


class ExtractionAgent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def call(self, user_prompt: str, response_model: type[BaseModel]) -> BaseModel:
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是一个文本翻译信息抽取助手。'
                    '你的任务是从用户输入中识别原始语种、目标语种和待翻译文本。'
                    '如果用户没有明确给出原始语种，请根据待翻译文本自动判断。'
                    '必须调用工具返回结构化结果。'
                ),
            },
            {'role': 'user', 'content': user_prompt},
        ]

        schema = response_model.model_json_schema()
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': schema['title'],
                    'description': '识别翻译请求中的原始语种、目标语种和待翻译文本。',
                    'parameters': {
                        'type': 'object',
                        'properties': schema['properties'],
                        'required': schema['required'],
                    },
                },
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice='auto',
        )

        arguments = response.choices[0].message.tool_calls[0].function.arguments
        return response_model.model_validate_json(arguments)


def translate_text(source_language: str, target_language: str, text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "你是一个翻译助手。请只返回翻译结果，不要添加解释。",
            },
            {
                "role": "user",
                "content": (
                    f"请把下面文本从{source_language}翻译成{target_language}：\n{text}"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    agent = ExtractionAgent()
    demo_prompts = [
        '帮我将英语good翻译为中文',
        '帮我将法语bon翻译为中文',
        '帮我将西班牙语bueno翻译为中文',
        '帮我将德语gut翻译为中文',
        '帮我将日语いい翻译为中文',
        '帮我将韩语좋아요翻译为中文',
    ]

    for user_prompt in demo_prompts:
        result = agent.call(user_prompt, TranslationTask)
        translation = translate_text(
            result.source_language,
            result.target_language,
            result.text,
        )

        print(f'用户输入：{user_prompt}')
        print('识别结果：')
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        print('翻译结果：')
        print(translation)
        print('-' * 40)
