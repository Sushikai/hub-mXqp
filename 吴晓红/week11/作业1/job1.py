# https://github.com/openai/openai-agents-python
# pip install -qU openai-agents
# pip install "openai-agents[viz]"

import os
from fastapi import FastAPI
from pydantic import BaseModel
from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled

# Pydantic models for request/response
class GroupRequest(BaseModel):
    """Request model for grouping operations"""
    msg: str

class GroupResponse(BaseModel):
    """Response model for grouping operations"""
    result: str
    message: str

app = FastAPI(
    title="Group API",
    description="API for grouping operations with AI agent integration",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI endpoint
    redoc_url="/redoc",  # ReDoc endpoint
)

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
agent = Agent(
    model="qwen-max",  # 模型代号
    name="Assistant",  # 给agent的取得名字（推荐英文，写的有意义）
    instructions="You are a helpful assistant"  # 对话中的 开头 system message
)
#
# result = Runner.run_sync(self.agent, "帮我写一个对联。")  # 同步运行，输入 user messgae
# print(result.final_output)


@app.post("/groupByObj", response_model=GroupResponse)
def groupByObj(request: GroupRequest):
    prompt='属于以下什么实体识别: 动物、植物、时间、地点'
    msg = f"{prompt}\n{request.msg}"
    result = Runner.run_sync(agent, msg)  # 同步运行，输入 user messgae
    print(result.final_output)
    return GroupResponse(result=result.final_output, message=f"Object grouping result for: {request.msg}")


@app.post("/groupByFeeling", response_model=GroupResponse)
def groupByFeeling(request: GroupRequest):
    prompt='属于以下什么类型情感分类: 高兴、悲伤、焦虑、惊讶'
    msg = f"{prompt}\n{request.msg}"
    result = Runner.run_sync(agent, msg)  # 同步运行，输入 user messgae
    print(result.final_output)
    return GroupResponse(result=result.final_output, message=f"Feeling grouping result for: {request.msg}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
