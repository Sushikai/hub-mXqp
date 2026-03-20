from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import Literal
import openai

client = openai.OpenAI(
    api_key="sk-f0abuabu58044adcb75b5a60974549b3",  # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TranslationAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自动生成tool格式
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
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Translation(BaseModel):
    """文本翻译信息提取"""
    source_text: str = Field(description="需要翻译的原始文本内容")
    source_language: str = Field(description="原始语种，如：英语、中文、日语、法语、德语等")
    target_language: str = Field(description="目标语种，如：中文、英语、日语、法语、德语等")
    translation_result: Optional[str] = Field(description="翻译结果（可选）", default=None)


# 测试案例
print("="*50)
print("测试1：英语翻译成中文")
print("="*50)
result = TranslationAgent(model_name="qwen-plus").call('帮我将good！翻译为中文', Translation)
print(f"原始文本: {result.source_text}")
print(f"原始语种: {result.source_language}")
print(f"目标语种: {result.target_language}")
print(f"翻译结果: {result.translation_result}")

print("\n" + "="*50)
print("测试2：中文翻译成英语")
print("="*50)
result = TranslationAgent(model_name="qwen-plus").call('请把"你好，世界"翻译成英文', Translation)
print(f"原始文本: {result.source_text}")
print(f"原始语种: {result.source_language}")
print(f"目标语种: {result.target_language}")

print("\n" + "="*50)
print("测试3：日语翻译成中文")
print("="*50)
result = TranslationAgent(model_name="qwen-plus").call('将"こんにちは"翻译成中文', Translation)
print(f"原始文本: {result.source_text}")
print(f"原始语种: {result.source_language}")
print(f"目标语种: {result.target_language}")

print("\n" + "="*50)
print("测试4：复杂翻译需求")
print("="*50)
result = TranslationAgent(model_name="qwen-plus").call('你能帮我把"What is your name?"这句话翻译成法语吗？', Translation)
print(f"原始文本: {result.source_text}")
print(f"原始语种: {result.source_language}")
print(f"目标语种: {result.target_language}")


class EnhancedTranslation(BaseModel):
    """增强版文本翻译信息提取"""
    source_text: str = Field(description="需要翻译的原始文本内容")
    source_language_code: str = Field(description="原始语种代码，如：en, zh, ja, fr, de等")
    source_language_name: str = Field(description="原始语种名称，如：英语、中文、日语等")
    target_language_code: str = Field(description="目标语种代码，如：en, zh, ja, fr, de等")
    target_language_name: str = Field(description="目标语种名称，如：英语、中文、日语等")
    translation_result: Optional[str] = Field(description="翻译结果", default=None)


# 测试增强版
print("\n" + "="*50)
print("增强版测试")
print("="*50)
result = TranslationAgent(model_name="qwen-plus").call('请把"hello world"翻译成中文', EnhancedTranslation)
print(f"原始文本: {result.source_text}")
print(f"原始语种: {result.source_language_name} ({result.source_language_code})")
print(f"目标语种: {result.target_language_name} ({result.target_language_code})")
