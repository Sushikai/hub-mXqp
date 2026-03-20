from pydantic import BaseModel, Field
from typing import Literal
import openai

# 初始化OpenAI客户端（兼容阿里云通义千问）
client = openai.OpenAI(
    api_key="sk-f0abuabu58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    """通用信息抽取智能体（抽取翻译所需信息）"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        # 增强Prompt，引导大模型抽取翻译信息
        enhanced_prompt = f"""
        请你作为翻译信息抽取助手，从用户指令中精准提取以下信息：
        1. 待翻译文本的原始语种（仅从【中文、英文、日语、法语、德语】中选择）
        2. 需要翻译成的目标语种（仅从【中文、英文、日语、法语、德语】中选择）
        3. 待翻译的文本内容
        用户指令：{user_prompt}
        请严格按照指定的工具参数格式返回结果，不要返回其他无关内容。
        """

        messages = [{"role": "user", "content": enhanced_prompt}]

        # 自定义工具定义（清晰的名称和描述）
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_translation_info",
                    "description": "抽取文本翻译所需的核心信息：原始语种、目标语种、待翻译文本",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "original_language": {
                                "type": "string",
                                "description": "待翻译文本的原始语种，只能是中文/英文/日语/法语/德语中的一个",
                                "enum": ["中文", "英文", "日语", "法语", "德语"]
                            },
                            "target_language": {
                                "type": "string",
                                "description": "需要翻译成的目标语种，只能是中文/英文/日语/法语/德语中的一个",
                                "enum": ["中文", "英文", "日语", "法语", "德语"]
                            },
                            "text_to_translate": {
                                "type": "string",
                                "description": "需要被翻译的具体文本内容，保留原始格式"
                            }
                        },
                        "required": ["original_language", "target_language", "text_to_translate"],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_translation_info"}},
            temperature=0.1,
        )

        try:
            if not response.choices[0].message.tool_calls:
                print(f"大模型未调用工具，原始响应：{response.choices[0].message.content}")
                return None
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(f'抽取失败：{e}，响应信息：{response.choices[0].message}')
            return None


class TranslationAgent:
    """翻译执行智能体（基于抽取的信息完成翻译）"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def translate(self, translation_info):
        """
        执行翻译
        :param translation_info: TranslationInfo对象（包含原始语种、目标语种、待翻译文本）
        :return: 翻译后的文本
        """
        # 构造翻译Prompt，明确翻译要求
        translate_prompt = f"""
        请你作为专业翻译助手，完成以下翻译任务：
        - 原始语种：{translation_info.original_language}
        - 目标语种：{translation_info.target_language}
        - 待翻译文本：{translation_info.text_to_translate}

        要求：
        1. 翻译结果准确、通顺，符合目标语种的表达习惯；
        2. 仅返回翻译后的文本，不要添加任何额外解释或说明。
        """

        messages = [{"role": "user", "content": translate_prompt}]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # 低随机性保证翻译准确性
            )
            # 提取翻译结果
            translated_text = response.choices[0].message.content.strip()
            return translated_text
        except Exception as e:
            print(f"翻译失败：{e}")
            return None


# 定义翻译信息抽取的Pydantic模型
class TranslationInfo(BaseModel):
    """文本翻译信息解析（识别原始语种、目标语种、待翻译文本）"""
    original_language: Literal["中文", "英文", "日语", "法语", "德语"] = Field(
        description="待翻译文本的原始语种"
    )
    target_language: Literal["中文", "英文", "日语", "法语", "德语"] = Field(
        description="需要翻译成的目标语种"
    )
    text_to_translate: str = Field(
        description="需要被翻译的文本内容"
    )


# 整合抽取+翻译的完整流程
def full_translation_pipeline(user_prompt, model_name="qwen-plus"):
    """
    完整的翻译流程：
    1. 抽取翻译信息（原始语种、目标语种、待翻译文本）
    2. 执行翻译
    3. 返回完整结果
    """
    # 步骤1：初始化抽取智能体并抽取信息
    extract_agent = ExtractionAgent(model_name)
    translation_info = extract_agent.call(user_prompt, TranslationInfo)
    if not translation_info:
        return {"status": "failed", "message": "无法抽取翻译信息"}

    # 步骤2：初始化翻译智能体并执行翻译
    translate_agent = TranslationAgent(model_name)
    translated_text = translate_agent.translate(translation_info)
    if not translated_text:
        return {"status": "failed", "message": "翻译执行失败"}

    # 步骤3：返回整合结果
    return {
        "status": "success",
        "extracted_info": {
            "original_language": translation_info.original_language,
            "target_language": translation_info.target_language,
            "text_to_translate": translation_info.text_to_translate
        },
        "translated_text": translated_text
    }


# 测试完整流程
if __name__ == "__main__":
    # 测试案例1：英文转中文
    prompt = "帮我将good！翻译为中文"
    result = full_translation_pipeline(prompt)
    if result["status"] == "success":
        print(f"原始语种：{result['extracted_info']['original_language']}")
        print(f"目标语种：{result['extracted_info']['target_language']}")
        print(f"待翻译文本：{result['extracted_info']['text_to_translate']}")
        print(f"翻译结果：{result['translated_text']}")
