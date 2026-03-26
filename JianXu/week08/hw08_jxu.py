

# -*- coding: utf-8 -*-
"""
文本翻译智能体
功能：
1. 自动识别用户请求中的：
   - 原始语种
   - 目标语种
   - 待翻译文本
2. 再调用翻译流程，输出翻译结果

"""

from pydantic import BaseModel, Field
from typing import Optional
import openai
import json
import os


# =========================
# 1. 初始化大模型客户端
# =========================

client = openai.OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") if None else "sk-f0abuabu58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# =========================
# 2. 定义信息抽取的数据模型
# =========================
class TranslationTask(BaseModel):
    """从用户请求中抽取翻译任务关键信息"""
    source_language: str = Field(description="原始语种，例如：英文、中文、日文、法文")
    target_language: str = Field(description="目标语种，例如：中文、英文、日文、法文")
    text: str = Field(description="待翻译的原始文本内容")


# =========================
# 3. 定义翻译结果的数据模型
# =========================
class TranslationResult(BaseModel):
    """翻译结果"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    original_text: str = Field(description="原始文本")
    translated_text: str = Field(description="翻译后的文本")


# =========================
# 4. 通用抽取智能体
# =========================
class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call(self, user_prompt: str, response_model):
        """
        根据 response_model 自动生成 tools 的 json schema，
        让模型按指定结构提取信息
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个结构化信息抽取助手。"
                    "你必须根据提供的工具定义，准确提取参数。"
                    "不要输出多余解释，只调用工具。"
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # 根据 Pydantic 模型自动构造 tool schema
        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema["title"],                  # 工具名
                    "description": schema.get("description", "信息抽取工具"),  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],   # 字段定义
                        "required": schema.get("required", []) # 必填字段
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
            # 取出模型生成的函数调用参数
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 转换为 Pydantic 对象并校验
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print("抽取失败：", e)
            print("原始返回：", response.choices[0].message)
            return None


# =========================
# 5. 翻译智能体
#    先抽取翻译任务，再执行翻译
# =========================
class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.extractor = ExtractionAgent(model_name=model_name)

    def extract_translation_task(self, user_request: str) -> Optional[TranslationTask]:
        """
        第一步：从用户请求中自动识别：
        - 原始语种
        - 目标语种
        - 待翻译文本
        """
        prompt = f"""
请从下面用户请求中抽取翻译任务信息。

要求：
1. 识别原始语种 source_language
2. 识别目标语种 target_language
3. 识别待翻译文本 text
4. 如果用户说“翻译为中文”，则 target_language = 中文
5. 如果待翻译文本是英文单词或英文句子，则 source_language = 英文
6. 如果待翻译文本是中文，则 source_language = 中文
7. 输出时只需按工具格式抽取，不要解释

用户请求：
{user_request}
"""
        return self.extractor.call(prompt, TranslationTask)

    def translate(self, task: TranslationTask) -> Optional[TranslationResult]:
        """
        第二步：根据抽取到的结构化信息执行翻译
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业翻译助手。"
                    "请严格按照用户指定的源语言和目标语言进行翻译。"
                    "只返回翻译后的结果，不要添加解释。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"请把下面内容从{task.source_language}翻译成{task.target_language}：\n"
                    f"{task.text}"
                )
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        translated_text = response.choices[0].message.content.strip()

        return TranslationResult(
            source_language=task.source_language,
            target_language=task.target_language,
            original_text=task.text,
            translated_text=translated_text
        )

    def run(self, user_request: str) -> Optional[TranslationResult]:
        """
        智能体主流程：
        1. 自动抽取翻译任务
        2. 执行翻译
        3. 返回完整结果
        """
        task = self.extract_translation_task(user_request)
        if not task:
            print("未能识别翻译任务")
            return None

        print("========== 第一步：自动识别翻译任务 ==========")
        print("原始语种：", task.source_language)
        print("目标语种：", task.target_language)
        print("待翻译文本：", task.text)

        result = self.translate(task)
        return result


# =========================
# 6. 主程序测试
# =========================
if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")

    # 示例输入
    user_request = "帮我将good！翻译为中文"

    result = agent.run(user_request)

    print("\n========== 第二步：翻译结果 ==========")
    if result:
        print("原始语种：", result.source_language)
        print("目标语种：", result.target_language)
        print("原始文本：", result.original_text)
        print("翻译结果：", result.translated_text)
    else:
        print("翻译失败")
