from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_NAME = "qwen-max"


class SentimentAnalysisOutput(BaseModel):
    """情感分类结果。"""

    task_type: Literal["sentiment"] = "sentiment"
    sentiment_label: Literal["积极", "中性", "消极"] = Field(
        description="文本情感分类标签。"
    )
    confidence: float = Field(description="0 到 1 之间的置信度。")
    reason: str = Field(description="对分类结果的简要说明。")


class EntityItem(BaseModel):
    """单个实体信息。"""

    text: str = Field(description="识别到的实体文本。")
    entity_type: str = Field(description="实体类型，例如 PERSON、LOCATION、ORG、TIME。")


class EntityExtractionOutput(BaseModel):
    """实体识别结果。"""

    task_type: Literal["entity"] = "entity"
    entities: list[EntityItem] = Field(default_factory=list, description="识别出的实体列表。")
    summary: str = Field(description="实体识别结果总结。")


def configure_runtime() -> str:
    """配置运行环境并返回模型名称。"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请先在终端中配置 API Key 后再运行本脚本。"
        )

    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    return os.getenv("HOMEWORK_MODEL", DEFAULT_MODEL_NAME)


def build_agents(model_name: str) -> Agent:
    """构建主 Agent 和两个子 Agent。"""

    sentiment_agent = Agent(
        name="SentimentAgent",
        model=model_name,
        handoff_description="负责对文本进行情感分类。",
        instructions=(
            "你是一个文本情感分类专家。"
            "当用户要求你做情感分类时，请判断文本属于积极、中性还是消极，"
            "并返回结构化结果。置信度必须在 0 到 1 之间。"
        ),
        output_type=SentimentAnalysisOutput,
    )

    entity_agent = Agent(
        name="EntityAgent",
        model=model_name,
        handoff_description="负责对文本进行实体识别。",
        instructions=(
            "你是一个文本实体识别专家。"
            "当用户要求你做实体识别时，请提取文本中的人名、地点、组织、时间等实体，"
            "仅保留文本中明确出现的实体，并返回结构化结果。"
        ),
        output_type=EntityExtractionOutput,
    )

    return Agent(
        name="HomeworkRouterAgent",
        model=model_name,
        instructions=(
            "你是一个任务路由主 Agent。"
            "如果用户要求做情感分类，就交给 SentimentAgent；"
            "如果用户要求做实体识别，就交给 EntityAgent。"
            "你只负责选择最合适的子 Agent，不直接完成分类或识别。"
        ),
        handoffs=[sentiment_agent, entity_agent],
    )


def format_result_payload(result_payload: SentimentAnalysisOutput | EntityExtractionOutput) -> str:
    """将结构化结果格式化为便于阅读的 JSON 字符串。"""

    return json.dumps(
        result_payload.model_dump(),
        ensure_ascii=False,
        indent=2,
    )


async def run_demo_case(router_agent: Agent, title: str, query: str) -> None:
    """运行单个固定案例并打印结果。"""

    print("\n" + "=" * 80)
    print(f"案例名称: {title}")
    print(f"原始请求: {query}")

    result = await Runner.run(router_agent, query)
    last_agent_name = result.last_agent.name
    print(f"主 Agent 路由结果: {last_agent_name}")

    if isinstance(result.final_output, SentimentAnalysisOutput):
        print("最终任务类型: 情感分类")
        print("结构化结果:")
        print(format_result_payload(result.final_output))
        return

    if isinstance(result.final_output, EntityExtractionOutput):
        print("最终任务类型: 实体识别")
        print("结构化结果:")
        print(format_result_payload(result.final_output))
        return

    raise TypeError(f"未识别的输出类型: {type(result.final_output)!r}")


async def main() -> None:
    """运行固定案例演示。"""

    model_name = configure_runtime()
    router_agent = build_agents(model_name)

    demo_cases = [
        (
            "情感分类示例 1",
            "请对下面这段文本做情感分类：这个新版本界面很漂亮，功能也很实用，我非常喜欢。",
        ),
        (
            "情感分类示例 2",
            "请对下面这段文本做情感分类：客服回复太慢了，系统还总是报错，这次体验让我很失望。",
        ),
        (
            "实体识别示例 1",
            "请做实体识别：马云在杭州创办了阿里巴巴集团。",
        ),
        (
            "实体识别示例 2",
            "请做实体识别：2026年4月，OpenAI 团队在旧金山发布了新的开发工具。",
        ),
    ]

    print("开始运行作业 1 固定案例演示...")
    print(f"当前模型: {model_name}")

    for title, query in demo_cases:
        await run_demo_case(router_agent, title, query)

    print("\n所有固定案例运行完成。")


if __name__ == "__main__":
    asyncio.run(main())

