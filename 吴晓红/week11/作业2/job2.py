#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业职能助手 - 自定义工具演示
实现3个自定义工具，支持自然语言对话调用
"""

import asyncio
import os
from typing import Annotated
from datetime import datetime
import math
import agents

# 设置API密钥（请替换为您自己的阿里云DashScope API密钥）
os.environ["OPENAI_API_KEY"] = "sk-"  # 示例key，请替换为您自己的
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云DashScope端点

from agents import Agent, Runner, function_tool, Tool, AsyncOpenAI, OpenAIChatCompletionsModel, set_default_openai_api, set_tracing_disabled

# 配置Agent设置（参考steamlit_demo.py）
# set_default_openai_api("chat_completions")  # 已通过环境变量和自定义客户端配置
set_tracing_disabled(True)  # 禁用跟踪以避免Tracing client错误

# ============================================================================
# 1. 自定义工具定义
# ============================================================================

@function_tool
def calculate_expression(
    expression: Annotated[str, "数学表达式，例如：2+3*4, sqrt(16), sin(30)"]
) -> str:
    """
    计算数学表达式。
    支持基本运算符：+ - * / % **
    支持函数：sqrt, sin, cos, tan, log, log10, pi, e
    示例：计算 "2+3*4", "sqrt(16)", "sin(30)"
    """
    try:
        # 定义安全的数学环境
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10,
            'pi': math.pi, 'e': math.e,
            'radians': math.radians, 'degrees': math.degrees
        }
        
        # 转换角度到弧度（如果输入的是度数）
        expr = expression.lower().replace('sin(', 'math.sin(math.radians(')
        expr = expr.replace('cos(', 'math.cos(math.radians(')
        expr = expr.replace('tan(', 'math.tan(math.radians(')
        
        # 恢复其他函数名
        expr = expr.replace('math.sqrt', 'sqrt').replace('math.log', 'log').replace('math.log10', 'log10')
        
        # 替换乘方符号
        expr = expr.replace('^', '**')
        
        # 评估表达式
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        
        # 格式化结果
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 6)
        
        return f"{expression} = {result}"
        
    except Exception as e:
        return f"计算错误：{e}。请检查表达式格式。"

@function_tool
def get_current_time(
    timezone: Annotated[str, "时区，例如：'上海', 'UTC', '纽约', '伦敦'"] = "上海"
) -> str:
    """
    获取指定时区的当前时间。
    支持时区：上海/北京（东八区）、UTC、纽约（西五区）、伦敦（格林威治时间）
    """
    try:
        # 获取当前UTC时间
        # utc_now = datetime.utcnow()
        utc_now = datetime.now()
        # 时区偏移（小时）
        timezone_offsets = {
            '上海': 8, '北京': 8, '中国': 8, '上海时间': 8, '北京时间': 8,
            'utc': 0, '格林威治': 0, '伦敦': 0,
            '纽约': -5, '美国东部': -5,
            '东京': 9, '日本': 9,
            '巴黎': 1, '法国': 1,
            '柏林': 1, '德国': 1
        }
        
        # 查找时区偏移
        tz_lower = timezone.lower()
        offset = 8  # 默认东八区（上海）
        
        for tz_key, tz_offset in timezone_offsets.items():
            if tz_key in tz_lower:
                offset = tz_offset
                break
        
        # 计算本地时间
        local_time = utc_now.hour + offset
        if local_time >= 24:
            local_time -= 24
            day_offset = 1
        elif local_time < 0:
            local_time += 24
            day_offset = -1
        else:
            day_offset = 0
        
        # 格式化时间
        time_str = f"{local_time:02d}:{utc_now.minute:02d}:{utc_now.second:02d}"
        
        # 添加日偏移说明
        day_info = ""
        if day_offset > 0:
            day_info = "（明天）"
        elif day_offset < 0:
            day_info = "（昨天）"
        
        return f"{timezone}当前时间：{time_str}{day_info}（UTC{offset:+d}时区）"
        
    except Exception as e:
        return f"获取时间错误：{e}"

@function_tool
def company_employee_info(
    query: Annotated[str, "查询内容，例如：'张三', '技术部', '所有员工'"]
) -> str:
    """
    查询公司员工信息。
    可以按姓名查询，按部门查询，或列出所有员工。
    示例：查询张三的信息、技术部有哪些员工、列出所有员工
    """
    # 模拟的员工数据库
    employees = [
        {"name": "张三", "department": "技术部", "position": "软件工程师", "email": "zhangsan@company.com"},
        {"name": "李四", "department": "技术部", "position": "前端工程师", "email": "lisi@company.com"},
        {"name": "王五", "department": "市场部", "position": "市场经理", "email": "wangwu@company.com"},
        {"name": "赵六", "department": "人事部", "position": "HR专员", "email": "zhaoliu@company.com"},
        {"name": "钱七", "department": "财务部", "position": "财务总监", "email": "qianqi@company.com"},
        {"name": "孙八", "department": "技术部", "position": "架构师", "email": "sunba@company.com"},
    ]
    
    query_lower = query.lower()
    
    # 1. 查询所有员工
    if "所有" in query_lower or "全部" in query_lower or "list" in query_lower:
        result = "公司员工列表：\n"
        for i, emp in enumerate(employees, 1):
            result += f"{i}. {emp['name']} - {emp['department']} - {emp['position']} ({emp['email']})\n"
        return result
    
    # 2. 按部门查询
    departments = ["技术部", "市场部", "人事部", "财务部"]
    for dept in departments:
        if dept in query:
            dept_emps = [emp for emp in employees if emp["department"] == dept]
            if dept_emps:
                result = f"{dept}员工：\n"
                for i, emp in enumerate(dept_emps, 1):
                    result += f"{i}. {emp['name']} - {emp['position']} ({emp['email']})\n"
                return result
    
    # 3. 按姓名查询
    for emp in employees:
        if emp["name"] in query:
            return f"员工信息：\n姓名：{emp['name']}\n部门：{emp['department']}\n职位：{emp['position']}\n邮箱：{emp['email']}"
    
    # 4. 模糊匹配
    matched = []
    for emp in employees:
        if query_lower in emp["name"].lower() or query_lower in emp["department"].lower() or query_lower in emp["position"].lower():
            matched.append(emp)
    
    if matched:
        result = f"找到{len(matched)}位匹配的员工：\n"
        for i, emp in enumerate(matched, 1):
            result += f"{i}. {emp['name']} - {emp['department']} - {emp['position']}\n"
        return result
    
    return f"未找到与'{query}'相关的员工信息。"

# ============================================================================
# 2. 创建Agent
# ============================================================================

def create_agent():
    """
    创建带有自定义工具的Agent，配置阿里云DashScope API
    """
    # 获取API密钥和端点
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 创建阿里云DashScope客户端
    dashscope_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # 将所有工具放入列表
    custom_tools: list[Tool] = [calculate_expression, get_current_time, company_employee_info]
    
    # 创建Agent，使用阿里云DashScope模型
    agent = Agent(
        name="企业职能助手",
        instructions="""
        你是一个企业内部职能助手，可以帮助员工处理各种查询和任务。
        你有以下工具可以使用：
        1. calculate_expression - 计算数学表达式
        2. get_current_time - 获取指定时区的当前时间
        3. company_employee_info - 查询公司员工信息
        
        当用户的问题需要用到工具时，请选择合适的工具并调用。
        如果用户的问题不需要工具，请直接回答。
        回答时请保持友好、专业的语气。
        """,
        tools=custom_tools,
        model=OpenAIChatCompletionsModel(
            model="qwen-max",  # 使用qwen-max模型，也可以改为qwen-plus或qwen-turbo
            openai_client=dashscope_client,
        ),
    )
    
    return agent

# ============================================================================
# 3. 对话循环
# ============================================================================

async def chat_loop():
    """
    主对话循环
    """
    print("=" * 60)
    print("企业职能助手 - 自定义工具演示")
    print("=" * 60)
    print("可用的功能：")
    print("  1. 数学计算（例如：计算 2+3*4, sqrt(16), sin(30)）")
    print("  2. 时间查询（例如：现在上海时间几点，UTC时间）")
    print("  3. 员工信息查询（例如：查询张三的信息，技术部有哪些员工）")
    print("  4. 其他问题（例如：你好，你是谁，你能做什么）")
    print("输入 '退出' 或 'quit' 结束对话")
    print("=" * 60)
    print()
    
    # 创建Agent
    agent = create_agent()
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n您：").strip()
            
            if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                print("助手：再见！")
                break
                
            if not user_input:
                continue
            
            print("助手：", end="", flush=True)
            
            # 运行Agent并获取结果
            result = await Runner.run(agent, input=user_input)
            
            # 输出结果
            if result.final_output:
                print(result.final_output)
            else:
                print("抱歉，我没有得到有效的回复。")
        except Exception as e:
            print(f"\n发生错误：{e}")

# ============================================================================
# 4. 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("正在启动企业职能助手...")
    try:
        # 运行对话循环
        asyncio.run(chat_loop())
    except Exception as e:
        print(f"\n启动失败：{e}")
