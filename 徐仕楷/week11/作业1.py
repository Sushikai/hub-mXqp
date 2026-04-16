import os
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled

# ====================== 配置日志 ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================== 配置环境变量 ======================
def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-":
        raise ValueError("OPENAI_API_KEY 环境变量未正确设置")
    return api_key


# ====================== Pydantic Models ======================
class GroupRequest(BaseModel):
    """请求模型"""
    msg: str = Field(..., min_length=1, max_length=500, description="需要分类的文本")


class GroupResponse(BaseModel):
    """响应模型"""
    result: str
    message: str
    category: str  # 新增：返回使用的分类类型，便于前端理解


# ====================== 分类类型定义 ======================
EntityType = Literal["obj", "feeling"]


# ====================== Agent 初始化 ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    try:
        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)

        global agent
        agent = Agent(
            model="qwen-max",
            name="Classifier",
            instructions=(
                "你是一个专业的文本分类助手。请严格按照用户指定的分类类别进行判断，"
                "只返回最匹配的一个类别，不要解释，不要输出多余内容。"
            )
        )
        logger.info("AI Agent 初始化成功，使用模型: qwen-max")
    except Exception as e:
        logger.error(f"Agent 初始化失败: {e}")
        raise
    yield
    # 关闭时清理（可选）


app = FastAPI(
    title="智能分类 API",
    description="基于阿里云通义千问的文本实体与情感分类服务",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 全局 Agent（通过 lifespan 初始化）
agent: Agent = None  # type: ignore


# ====================== 依赖注入 ======================
def get_agent() -> Agent:
    if agent is None:
        raise HTTPException(status_code=500, detail="AI Agent 未初始化")
    return agent


# ====================== 通用分类函数 ======================
async def classify_text(
        text: str,
        category_type: EntityType,
        agent: Agent
) -> str:
    """通用分类逻辑"""
    if category_type == "obj":
        prompt = "请将以下内容归类到最匹配的实体类型，只能返回以下之一：动物、植物、时间、地点"
    else:  # feeling
        prompt = "请将以下内容归类到最匹配的情感类型，只能返回以下之一：高兴、悲伤、焦虑、惊讶"

    msg = f"{prompt}\n\n文本：