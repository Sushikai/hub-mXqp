from pydantic import BaseModel, Field
from typing import List, Optional  # ← 只在这里多加了 Optional
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-XXXX",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


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
        # 传入需要提取的内容，自己写了一个tool格式
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

        if not response.choices[0].message.tool_calls:
            print("模型没有生成 tool call")
            print("模型直接回答：", response.choices[0].message.content)
            return None

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

class TransLateAgent(BaseModel):
    """文本问答内容解析"""
    source_lang: Optional[str] = Field(  # 加了 Optional + default
        default=None,
        description="源语言，可以是中文名称或代码，如：中文、英语、日语、en、zh、日文 等。如果未明确说明可为空，由模型自动检测"
    )
    target_lang: str = Field(
        description="目标语言，必须明确，如：中文、英语、日语、韩语、法语、德语、en、zh、ja 等"
    )
    text: str = Field(
        description="需要被翻译的**原始文本**（请完整保留原文，不要提前翻译、不要改写、不要省略标点）"
    )


if __name__ == "__main__":
    result = ExtractionAgent(model_name="qwen-plus").call(
        '帮我翻译成日语：你把我当日本人整',
        TransLateAgent
    )
    print(result)

    if result is not None and hasattr(result, 'text') and hasattr(result, 'target_lang'):
        print("\n=== 翻译结果 ===")

        prompt = f"把下面这段中文翻译成{result.target_lang}，只输出翻译结果，不要解释：\n{result.text}"

        trans_response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        translated = trans_response.choices[0].message.content.strip()
        print(translated)
    else:
        print("\n无法获取翻译参数")