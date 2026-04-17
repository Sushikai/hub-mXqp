#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业职能助手 - 优化版
支持数学计算、时区时间查询、公司员工信息查询
"""

import asyncio
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Annotated

from agents import (
    Agent,
    Runner,
    function_tool,
    Tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

# ====================== 配置日志 ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================== 环境变量配置 ======================
def load_config():
    """加载并验证配置"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key or api_key.startswith("sk-") and len(api_key) < 20:
        logger.warning("OPENAI_API_KEY 未正确设置，请通过环境变量提供有效密钥")

    return api_key, base_url


# ====================== 自定义工具 ======================
@function_tool
def calculate_expression(
        expression: Annotated[str, "要计算的数学表达式，例如：2+3*4, sqrt(16), sin(30°)"]
) -> str:
    """安全地计算数学表达式，支持基础运算和常见数学函数"""
    try:
        import math

        # 安全执行环境
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "pi": math.pi, "e": math.e,
            "radians": math.radians, "degrees": math.degrees,
            "pow": pow,
        }

        # 预处理表达式
        expr = expression.strip().lower()
        expr = expr.replace("^", "**")  # 支持 ^ 表示乘方
        expr = expr.replace("°", "")  # 移除度数符号

        # 安全评估
        result = eval(expr, {"__builtins__": {}}, safe_dict)

        # 格式化输出
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 6)

        return f"{expression} = {result}"

    except Exception as e:
        logger.warning(f"表达式计算失败: {expression} | 错误: {e}")
        return f"计算错误：{e}。请检查表达式格式（支持 + - * / ** sqrt sin cos tan 等）"


@function_tool
def get_current_time(
        timezone: Annotated[str, "时区名称，例如：上海、北京、UTC、纽约、东京、伦敦"] = "上海"
) -> str:
    """获取指定时区的当前时间（使用标准时区数据库）"""
    try:
        tz_map = {
            "上海": "Asia/Shanghai",
            "北京": "Asia/Shanghai",
            "中国": "Asia/Shanghai",
            "东京": "Asia/Tokyo",
            "日本": "Asia/Tokyo",
            "纽约": "America/New_York",
            "美国东部": "America/New_York",
            "伦敦": "Europe/London",
            "巴黎": "Europe/Paris",
            "柏林": "Europe/Berlin",
            "utc": "UTC",
            "格林威治": "UTC",
        }

        tz_name = tz_map.get(timezone.strip(), timezone.strip())

        # 如果不在映射表中，尝试直接使用用户输入作为时区
        try:
            now = datetime.now(ZoneInfo(tz_name))
        except Exception:
            # 回退到默认上海时间
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            tz_name = "Asia/Shanghai"

        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        return f"{timezone} 当前时间：{time_str}（{tz_name}）"

    except Exception as e:
        logger.error(f"获取时间失败: {e}")
        return f"获取时间失败：{e}"


@function_tool
def company_employee_info(
        query: Annotated[str, "查询关键词，例如：张三、技术部、所有员工"]
) -> str:
    """查询公司员工信息（模拟数据库）"""
    employees = [
        {"name": "张三", "department": "技术部", "position": "软件工程师", "email": "zhangsan@company.com"},
        {"name": "李四", "department": "技术部", "position": "前端工程师", "email": "lisi@company.com"},
        {"name": "王五", "department": "市场部", "position": "市场经理", "email": "wangwu@company.com"},
        {"name": "赵六", "department": "人事部", "position": "HR专员", "email": "zhaoliu@company.com"},
        {"name": "钱七", "department": "财务部", "position": "财务总监", "email": "qianqi@company.com"},
        {"name": "孙八", "department": "技术部", "position": "架构师", "email": "sunba@company.com"},
    ]

    q = query.strip().lower()

    # 查询所有员工
    if any(k in q for k in ["所有", "全部", "list", "全部员工"]):
        result = "公司全体员工列表：\n"
        for i, emp in enumerate(employees, 1):
            result += f"{i}. {emp['name']} | {emp['department']} | {emp['position']} | {emp['email']}\n"
        return result

    # 按部门查询
    for emp in employees:
        if emp["department"] in query:
            dept_emps = [e for e in employees if e["department"] == emp["department"]]
            result = f"{emp['department']} 员工列表：\n"
            for i, e in enumerate(dept_emps, 1):
                result += f"{i}. {e['name']} - {e['position']} ({e['email']})\n"
            return result

    # 按姓名精确匹配
    for emp in employees:
        if emp["name"] in query or emp["name"].lower() in q:
            return (f"员工信息：\n"
                    f"姓名：{emp['name']}\n"
                    f"部门：{emp['department']}\n"
                    f"职位：{emp['position']}\n"
                    f"邮箱：{emp['email']}")

    # 模糊匹配
    matched = [
        emp for emp in employees
        if q in emp["name"].lower() or q in emp["department"].lower() or q in emp["position"].lower()
    ]

    if matched:
        result = f"找到 {len(matched)} 条匹配记录：\n"
        for i, emp in enumerate(matched, 1):
            result += f"{i}. {emp['name']} - {emp['department']} - {emp['position']}\n"
        return result

    return f"未找到与 “{query}” 相关的信息。请尝试查询姓名、部门或输入“所有员工”。"


# ====================== 创建 Agent ======================
def create_agent() -> Agent:
    """创建并配置企业职能助手 Agent"""
    api_key, base_url = load_config()

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    tools: list[Tool] = [
        calculate_expression,
        get_current_time,
        company_employee_info,
    ]

    agent = Agent(
        name="企业职能助手",
        instructions="""
        你是一个专业的企业内部职能助手。
        你可以帮助用户进行数学计算、查询各时区当前时间、以及查询公司员工信息。

        使用工具时的原则：
        - 需要计算数学表达式时，使用 calculate_expression
        - 需要查询时间时，使用 get_current_time
        - 需要查询员工信息时，使用 company_employee_info
        - 其他一般问题可以直接回答

        请保持回答友好、专业、简洁。如果不确定，可以礼貌地询问澄清。
        """,
        tools=tools,
        model=OpenAIChatCompletionsModel(
            model="qwen-max",  # 可改为 qwen-plus / qwen-turbo
            openai_client=client,
        ),
    )

    logger.info("企业职能助手 Agent 初始化完成（模型：qwen-max）")
    return agent


# ====================== 对话循环 ======================
async def chat_loop():
    """交互式对话主循环"""
    print("=" * 70)
    print("               企业职能助手 - 自定义工具演示")
    print("=" * 70)
    print("支持功能：")
    print("  • 数学计算        →  计算 2+3*4、sqrt(16)、sin(30)")
    print("  • 时区时间查询    →  现在上海时间几点？纽约时间？")
    print("  • 员工信息查询    →  查询张三、技术部员工、所有员工")
    print("输入 '退出'、'quit' 或 'q' 结束对话")
    print("=" * 70)

    agent = create_agent()

    while True:
        try:
            user_input = input("\n您：").strip()

            if user_input.lower() in {"退出", "quit", "exit", "q"}:
                print("助手：再见！祝您工作愉快~")
                break

            if not user_input:
                continue

            print("助手：", end="", flush=True)

            result = await Runner.run(agent, input=user_input)

            print(result.final_output if result.final_output else "（无有效回复）")

        except KeyboardInterrupt:
            print("\n\n已取消对话，再见！")
            break
        except Exception as e:
            logger.error(f"对话过程中发生错误: {e}")
            print(f"\n发生错误：{e}")


# ====================== 主程序 ======================
if __name__ == "__main__":
    print("正在启动企业职能助手...\n")
    try:
        asyncio.run(chat_loop())
    except Exception as e:
        logger.critical(f"程序启动失败: {e}")
        print(f"启动失败：{e}")