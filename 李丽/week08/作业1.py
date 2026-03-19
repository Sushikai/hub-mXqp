import os
import openai
from pydantic import BaseModel, Field

# 默认读取环境变量中的 OPENAI_API_KEY 或者 ALIYUN_API_KEY
client = openai.OpenAI(
    api_key=os.environ.get("ALIYUN_API_KEY"), 
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
        except Exception as e:
            print('ERROR', e)
            return None

class Translation(BaseModel):
    """文本翻译参数提取"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    text_to_translate: str = Field(description="待翻译的文本")
    translated_text: str = Field(description="翻译后的文本结果")

def translate_text(agent: ExtractionAgent, text: str):
    print(f"正在提取并翻译: '{text}'...")
    result = agent.call(text, Translation)
    return result

if __name__ == '__main__':
    agent = ExtractionAgent(model_name="qwen-plus")
    
    # 在这里可以方便地修改测试数据
    test_cases = [
        "帮我将nice to meet you 翻译为中文",
        "把'黄金啥时候能涨价'翻译成英文",
        "将'Bonjour'翻译成日文"
    ]
    
    for case in test_cases:
        result = translate_text(agent, case)
        print("结果:", result)
        print("-" * 30)
