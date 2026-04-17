import asyncio
from typing import List

from agents import Agent, Runner
from pydantic import BaseModel


# 定义结构化输出模型
class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    explanation: str


class Entity(BaseModel):
    text: str
    type: str
    start_pos: int
    end_pos: int


class EntityResult(BaseModel):
    entities: List[Entity]
    summary: str


# 子 Agent 1: 情感分类
sentiment_agent = Agent(
    name="Sentiment Classifier",
    instructions="""
你是一个情感分析专家。请分析用户输入文本的情感。
输出要求:
1. sentiment: positive, negative 或 neutral
2. confidence: 0到1之间的置信度分数
3. explanation: 一句话解释判断理由
""",
    output_type=SentimentResult,
)

# 子 Agent 2: 实体识别
entity_agent = Agent(
    name="Entity Recognizer",
    instructions="""
你是一个命名实体识别专家。请从用户输入的文本中识别所有实体。
实体类型: PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT 等。
输出要求:
1. entities: 实体列表，每个实体包含 text, type, start_pos, end_pos (无法定位时填-1)
2. summary: 一句话总结识别出的实体类型
""",
    output_type=EntityResult,
)

# 主 Agent (路由)
triage_agent = Agent(
    name="Triage Agent",
    instructions="""
你是一个路由助手。判断用户意图并转交给最合适的专业Agent。
规则:
1. 情感分析、情绪判断 → 转交给 Sentiment Classifier
2. 实体识别、命名实体提取 → 转交给 Entity Recognizer
3. 意图不明确或同时包含两者 → 要求用户明确指定
注意: 只负责转交，不要自行回答专业问题。
""",
    handoffs=[sentiment_agent, entity_agent],
)


def format_sentiment(result: SentimentResult) -> str:
    sentiment_map = {"positive": "😊 正面", "negative": "😞 负面", "neutral": "😐 中性"}
    return (f"情感倾向: {sentiment_map.get(result.sentiment, result.sentiment)}\n"
            f"置信度: {result.confidence * 100:.1f}%\n"
            f"判断理由: {result.explanation}")


def format_entity(result: EntityResult) -> str:
    if not result.entities:
        return f"未识别到任何实体。\n\n总结: {result.summary}"
    output = f"识别到 {len(result.entities)} 个实体:\n"
    for e in result.entities:
        output += f"  • {e.text} ({e.type})\n"
    output += f"\n总结: {result.summary}"
    return output


async def main():
    print("=" * 60)
    print("多Agent系统 - 情感分析 + 实体识别")
    print("输入 'exit' 或 'quit' 退出")
    print("-" * 60)

    while True:
        user_input = input("\n请输入文本: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break
        if not user_input:
            continue

        try:
            result = await Runner.run(triage_agent, input=user_input)
            final = result.final_output

            if isinstance(final, SentimentResult):
                print("\n【情感分析结果】")
                print(format_sentiment(final))
            elif isinstance(final, EntityResult):
                print("\n【实体识别结果】")
                print(format_entity(final))
            else:
                print(f"\n【系统消息】\n{final}")
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
