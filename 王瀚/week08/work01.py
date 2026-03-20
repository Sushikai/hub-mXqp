import os
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal
import openai
import json

# ==========================================
# 1. 配置部分 (请替换为你的真实 API Key)
# ==========================================
# 【重要安全提示】生产环境中请使用 os.environ.get("DASHSCOPE_API_KEY")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ==========================================
# 2. 核心框架：ExtractionAgent (信息抽取智能体)
# ==========================================
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        """
        利用 Function Calling 机制，将用户自然语言转换为 Pydantic 模型对象
        """
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # 动态生成 Tool 定义
        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get('title', 'ExtractionTask'),
                    "description": schema.get('description', 'Extract structured data from text'),
                    "parameters": {
                        "type": "object",
                        "properties": schema.get('properties', {}),
                        "required": schema.get('required', []),
                    },
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # 让模型自动决定是否调用工具
            )

            # 检查是否有工具调用
            message = response.choices[0].message
            if message.tool_calls:
                arguments = message.tool_calls[0].function.arguments
                # 验证并转换为 Pydantic 对象
                return response_model.model_validate_json(arguments)
            else:
                print("警告：模型未调用工具，尝试直接解析...")
                # 兜底策略：如果模型没调工具，尝试直接把内容当 JSON 解析（视情况而定）
                return None

        except Exception as e:
            print(f'ERROR 抽取失败: {e}')
            print(f'原始响应: {response.choices[0].message if "response" in locals() else "No Response"}')
            return None


# ==========================================
# 3. 业务逻辑：翻译智能体
# ==========================================

class TranslationParams(BaseModel):
    """提取翻译任务的关键参数"""
    source_language: str = Field(
        description="原始语种。如果用户未明确指定，请根据待翻译文本自动推断（例如：'英语', '中文', '日语', '法语'）。"
    )
    target_language: str = Field(
        description="目标语种。用户希望翻译成的语言（例如：'中文', '英语', '西班牙语'）。"
    )
    text_to_translate: str = Field(
        description="需要被翻译的具体文本内容。请去除‘请翻译’、‘帮我’等指令性词汇，只保留纯净的原文。"
    )


class TranslationAgent:
    def __init__(self):
        self.extractor = ExtractionAgent(model_name=MODEL_NAME)

    def process(self, user_input: str):
        print(f"\n🚀 正在处理请求: '{user_input}'")
        print("-" * 30)

        # 第一步：抽取参数
        params = self.extractor.call(user_input, TranslationParams)

        if not params:
            print("❌ 无法提取翻译参数，请检查输入。")
            return None

        print(f"✅ [参数提取成功]")
        print(f"   🌍 源语言: {params.source_language}")
        print(f"   🎯 目标语言: {params.target_language}")
        print(f"   📝 待翻译文本: \"{params.text_to_translate}\"")
        print("-" * 30)

        # 第二步：执行翻译
        translation_result = self._execute_translation(params)

        if translation_result:
            print(f"💡 [翻译结果]:\n{translation_result}")
            return translation_result
        else:
            print("❌ 翻译执行失败。")
            return None

    def _execute_translation(self, params: TranslationParams):
        """调用大模型进行实际翻译"""
        messages = [
            {
                "role": "system",
                "content": f"你是一位专业的翻译专家。请将以下文本从 {params.source_language} 准确、流畅地翻译为 {params.target_language}。只输出翻译后的结果，不要包含任何解释、注音或额外格式。"
            },
            {
                "role": "user",
                "content": params.text_to_translate
            }
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3  # 翻译任务通常需要较低的创造性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"翻译 API 调用错误: {e}")
            return None


# ==========================================
# 4. 运行测试
# ==========================================
if __name__ == "__main__":
    agent = TranslationAgent()

    # 测试案例列表
    test_cases = [
        "帮我将 good！翻译为中文",
        "把 'Je t'aime' 翻成英文",
        "这句话 'こんにちは' 是什么意思？请把它翻译成中文。",
        "我需要将 'Artificial Intelligence' 译为中文，用于报告。",
        "Translate 'Hello World' to Japanese."
    ]

    for case in test_cases:
        agent.process(case)
        print("\n" + "=" * 50 + "\n")
