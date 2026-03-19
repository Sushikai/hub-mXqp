from pydantic import BaseModel, Field
import openai


# =========================
# 1. 初始化客户端
# =========================
client = openai.OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# =========================
# 2. 通用抽取智能体
# =========================
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

        # 根据 Pydantic 模型，自动生成 tool 描述
        schema = response_model.model_json_schema()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema["title"],              # 工具名
                    "description": schema.get("description", ""),  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema["required"],
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
            # 取出大模型生成的函数参数
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 转成 Pydantic 对象
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print("ERROR:", e)
            print("模型返回内容：", response.choices[0].message)
            return None


# =========================
# 3. 定义“翻译任务”数据结构
# =========================
class TranslateTask(BaseModel):
    """抽取翻译请求中的原始语种、目标语种和待翻译文本"""
    source_language: str = Field(description="原始语种，例如中文、英语、日语、法语等")
    target_language: str = Field(description="目标语种，例如中文、英语、日语、法语等")
    text: str = Field(description="待翻译的原始文本内容")


# =========================
# 4. 真正执行翻译的函数
# =========================
def translate_text(source_language: str, target_language: str, text: str, model_name="qwen-plus"):
    prompt = f"""
你是一个专业翻译助手。
请将下面的文本从{source_language}翻译成{target_language}。
只返回翻译结果，不要加解释。

待翻译文本：
{text}
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个专业、准确、简洁的翻译助手。"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# =========================
# 5. 封装成一个“翻译智能体”
# =========================
class TranslationAgent:
    def __init__(self, model_name="qwen-plus"):
        self.extractor = ExtractionAgent(model_name=model_name)
        self.model_name = model_name

    def run(self, user_input: str):
        # 第一步：抽取翻译任务参数
        task = self.extractor.call(user_input, TranslateTask)

        if task is None:
            return None

        # 第二步：执行翻译
        translated_result = translate_text(
            source_language=task.source_language,
            target_language=task.target_language,
            text=task.text,
            model_name=self.model_name
        )

        return {
            "source_language": task.source_language,
            "target_language": task.target_language,
            "text": task.text,
            "translated_text": translated_result
        }


# =========================
# 6. 测试
# =========================
if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")

    user_input = "帮我将good！翻译为中文"

    result = agent.run(user_input)

    print("抽取结果：")
    print("原始语种：", result["source_language"])
    print("目标语种：", result["target_language"])
    print("待翻译文本：", result["text"])
    print("翻译结果：", result["translated_text"])