import asyncio
from agents import Agent, Runner, trace

# 设置 API
# from agents import set_default_openai_api
# set_default_openai_api("chat_completions")

# 子 Agent 1: 情感分类
sentiment_agent = Agent(
    name="Sentiment Analyzer",
    instructions="""
    你是一个情感分析专家。请判断用户输入文本的情感倾向：
    - 正面（Positive）
    - 负面（Negative）
    - 中性（Neutral）
    只返回情感类别，不要解释。
    """,
)

# 子 Agent 2: 实体识别
ner_agent = Agent(
    name="Named Entity Recognizer",
    instructions="""
    你是一个命名实体识别（NER）专家。请从用户输入中提取以下类型的实体：
    - 人名（Person）
    - 地点（Location）
    - 组织（Organization）
    - 时间（Date/Time）
    以清晰格式列出识别到的实体，如果没有则说明“未识别到实体”。
    """,
)

# 主 Agent：路由选择
router_agent = Agent(
    name="Router Agent",
    instructions="""
    你是一个任务路由助手。根据用户输入判断任务类型：
    - 如果用户想分析情感（如“这句话感觉怎么样”、“情绪是正面还是负面”），调用 sentiment_agent。
    - 如果用户想提取实体（如“这段话提到了哪些人或地点”、“识别出人名和组织”），调用 ner_agent。
    不要自己回答，只选择正确的子 Agent 进行处理。
    """,
    handoffs=[sentiment_agent, ner_agent],
)

# 主程序
async def main():
    # 示例输入
    user_input = "苹果公司CEO库克访问了北京，市场反应非常积极。"

    print(f"用户输入: {user_input}\n")

    # 使用 trace 追踪整个流程（可选，用于调试）
    with trace("Agent Routing Workflow"):
        result = await Runner.run(router_agent, user_input)

    print(f"处理结果:\n{result.final_output}")
    print(f"最终处理 Agent: {result.last_agent.name}")

# 运行程序
if __name__ == "__main__":
    asyncio.run(main())
