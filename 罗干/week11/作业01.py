import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-b8d6efe8169b4351a91a13b8fe8fd99a"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 意图识别 -》 路由
# 用户提问 -》 类型1  类型2  类型3

sentiment_classification_agent = Agent(
    name="sentiment_classification_agent",
    model="qwen-max",
    instructions="你是一个语言情感分类专家，擅长对用户的文本进行情感分类，回答问题的时候先告诉我你擅长什么，输出的格式只从以下选项中选择[Happy,Angry,Neutral]",
)

entity_identification_agent = Agent(
    name="entity_identification_agent",
    model="qwen-max",
    instructions="你是一个语言实体识别专家，擅长对用户的文本进行实体识别，回答问题的时候先告诉我你擅长什么,输出格式为[XXX-人名;XXX-地名;XXX-时间;XXX-组织;XXX-货币]",
)


# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[sentiment_classification_agent, entity_identification_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])
    msg = input("你好，我可以帮你进行文本的情感分类，你还有什么问题？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())
