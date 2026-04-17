"""
1: 安装openai-agents框架，实现如下的一个程序：
• 有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
• 子agent 1: 对文本进行情感分类
• 子agent 2: 对文本进行实体识别
"""
import os
import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, trace, SQLiteSession, GuardrailFunctionOutput, InputGuardrail, \
    InputGuardrailTripwireTriggered
from agents import set_default_openai_api, set_tracing_disabled

os.environ["OPENAI_API_KEY"] = "sk-7458206891744b7aa46d6f7366fecdd5"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 在相同agent  或 不同agent 之间共享历史对话
session = SQLiteSession("conversation_123")

class ClassifyOutput(BaseModel):
    """用于判断用户的输入是否属于情感分类或实体识别"""
    is_scope: bool


guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions="判断用户的问题是否属于情感分类还是实体识别相关问题。如果是，'is_scope'应为 True， json 返回",
    output_type=ClassifyOutput,
)

# 情感分类agent
emotion_tutor_agent = Agent(
    name = "sentiment classification",
    model="qwen-max",
    handoff_description="对文本进行情感分类",
    instructions="请对文本进行情感分类，输出结果为情感类别（正面心情、负面心情，中性心情，未知心情）"
)

# 文本实体识别agent
entity_recognizer_agent = Agent(
    name = "entity identification",
    model="qwen-max",
    handoff_description="对文本进行实体识别",
    instructions="请对文本进行实体识别，识别出以下类型的实体：人名、组织名、地点、时间、事件、产品名等。以JSON格式返回结果，例如：{'entities': [{'text': '实体内容', 'type': '实体类型'}]}"
)

async def validate_input(ctx, agent, input_data):
    print(f"[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ClassifyOutput)

    tripwire_triggered = not final_output.is_scope

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )

# 主agent，接受用户的请求输入，选择其中的一个agent 回答
main_agent = Agent(
    name="主agent",
    model="qwen-max",
    instructions="您的任务是根据用户的请求输入，选择其中的一个agent 回答，输出结果为情感类别或实体列表",
    handoffs=[
        emotion_tutor_agent,
        entity_recognizer_agent
    ],
    input_guardrails = [
        InputGuardrail(guardrail_function=validate_input),
    ],
)


async def main():
    print("---启动情感分类，实体识别agent---")

    print("\n" + "=" * 50)
    try:
        query = "早上骑电动车上班不小心摔了一跤，感觉一天很糟糕"
        print(f"**用户输入:** {query}")
        result = await Runner.run(main_agent, query)
        print("**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    try:
        query = "下午好邻居超市新开业，我准备去买个行李箱"
        print(f"**用户输入:** {query}")
        result = await Runner.run(main_agent, query)
        print("**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    try:
        query = "请对以下文本进行实体识别：小浣熊方便面是康师傅公司2023年在北京推出的产品"
        print(f"**用户输入:** {query}")
        result = await Runner.run(main_agent, query)
        print("**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("**❌ 守卫阻断触发:**", e)

if __name__ == "__main__":
    asyncio.run(main())