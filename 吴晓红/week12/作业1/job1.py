# https://github.com/openai/openai-agents-python
# pip install -qU openai-agents
# pip install "openai-agents[viz]"

import os
from fastapi import FastAPI
from pydantic import BaseModel
from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled
import sqlite3

# Pydantic models for request/response
class GroupRequest(BaseModel):
    msg: str

class GroupResponse(BaseModel):
    msg: str

app = FastAPI(
    title="Group API",
    description="API for grouping operations with AI agent integration",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI endpoint
    redoc_url="/redoc",  # ReDoc endpoint
)
prompt='sqlite数据库，customers是用户表, employees是员工表，根据以下问题，生成一个可以直接执行的SQL查询语句，不要多余内容:'
os.environ["OPENAI_API_KEY"] = "sk-1b0891fe1ab844d98139f95bdd6b402b"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
agent = Agent(
    model="qwen3.5-flash",  # 模型代号
    name="Assistant",  # 给agent的取得名字（推荐英文，写的有意义）
    instructions="You are a helpful assistant"  # 对话中的 开头 system message
)

@app.post("/ask", response_model=GroupResponse)
def groupByObj(request: GroupRequest):
    msg = f"{prompt}  {request.msg}"
    result = Runner.run_sync(agent, msg)  # 同步运行，输入 user messgae
    print(result.final_output)
    conn = sqlite3.connect('chinook.db')
    cursor = conn.cursor()
    cursor.execute(result.final_output)
    tables = cursor.fetchall()
    print(tables)
    rsp = request.msg + '的答案是：' + str(tables)
    return GroupResponse(msg=rsp)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
