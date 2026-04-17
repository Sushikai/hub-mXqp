import os
import sys

import asyncio
import uuid

# 从环境变量获取 API Key
api_key = os.environ.get("OPENAI_API_KEY")

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 子agent 1: 对文本进行情感分类
sentiment_agent = Agent(
    name="sentiment_agent",
    model="qwen-max",
    instructions="你是情感分析专家，擅长对文本进行情感分类（正面、负面或中性）。回答问题时先告诉我你是情感分析专家，然后给出分类结果及原因。",
)

# 子agent 2: 对文本进行实体识别
ner_agent = Agent(
    name="ner_agent",
    model="qwen-max",
    instructions="你是实体识别专家，擅长对文本进行实体识别（如：人名、地名、组织机构名等）。回答问题时先告诉我你是实体识别专家，然后列出识别出的实体。",
)

# 主agent：接受用户请求并进行分发
main_agent = Agent(
    name="main_agent",
    model="qwen-max",
    instructions="你是一个主代理，负责将用户的请求转交给合适的专家处理。如果用户需要情感分析，请转交给sentiment_agent；如果需要实体识别，请转交给ner_agent。",
    handoffs=[sentiment_agent, ner_agent],
)

async def main():
    conversation_id = str(uuid.uuid4().hex[:16])
    print("=============================================")
    print("欢迎使用文本分析系统！（输入 'quit' 退出）")
    print("支持的功能：文本情感分类、文本实体识别")
    print("=============================================")
    
    agent = main_agent
    inputs: list[TResponseInputItem] = []
    
    while True:
        try:
            print("\n用户: ", end="", flush=True)
            # 使用 sys.stdin.readline() 读取并在 Windows 环境下安全解码
            user_msg = sys.stdin.buffer.readline().decode('utf-8', errors='ignore')
            if not user_msg:
                break
            user_msg = user_msg.strip()
        except EOFError:
            break
        if user_msg.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
            
        if not user_msg.strip():
            continue
            
        inputs.append({"content": user_msg, "role": "user"})
        
        print("Agent: ", end="")
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
                    # 直接打印，并在 print 中处理 flush
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    pass # Done with part
        print("\n")
        
        # 更新对话历史和当前处理的agent
        inputs = result.to_input_list()
        agent = result.current_agent

if __name__ == "__main__":
    asyncio.run(main())
