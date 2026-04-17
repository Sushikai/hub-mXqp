
"""根据项目需求，我们需要为企业职能助手增加三个自定义工具，用于查询员工年假余额、工资发放日以及最近存款日期。这些工具将集成到现有的 MCP Server 中，并通过 Streamlit 前端实现自然语言对话调用,
在 tool.py 中添加三个新工具函数，每个函数使用 @mcp.tool 装饰器，并提供清晰的名称、描述、参数类型及说明"""

import re
from typing import Annotated, Union
import requests
from datetime import date, timedelta

TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."],
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."],
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']

    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'

    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)

    count_positive = len(positive_matches)
    count_negative = len(negative_matches)

    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"

@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary based on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


# ========== 新增自定义工具 ==========

@mcp.tool
def query_annual_leave_balance(user_name: Annotated[str, "员工姓名（中文全名）"]):
    """
    查询指定员工的剩余年假天数。
    返回值为剩余年假天数（单位：天）。
    """
    # 模拟数据库或HR系统查询
    # 实际应替换为真实数据源（如 SQL 查询、API 调用）
    mock_data = {
        "张三": 12.5,
        "李四": 8.0,
        "王五": 0,
    }
    # 默认返回5天（演示）
    return mock_data.get(user_name, 5.0)


@mcp.tool
def query_salary_payment_date(user_name: Annotated[str, "员工姓名（中文全名）"]):
    """
    查询指定员工的每月工资发放日期。
    返回格式为字符串，例如：'每月15日' 或 '次月5日'。
    """
    # 模拟不同岗位/部门的发薪日差异
    mock_payment = {
        "张三": "每月15日",
        "李四": "每月20日",
        "王五": "次月5日",
    }
    # 默认值
    return mock_payment.get(user_name, "每月15日")


@mcp.tool
def query_last_deposit_date(user_name: Annotated[str, "员工姓名（中文全名）"]):
    """
    查询员工最近一笔存款的日期（金融客服场景）。
    返回日期字符串，格式：YYYY-MM-DD。
    """
    # 模拟银行系统或财务系统记录
    mock_last_deposit = {
        "张三": "2026-04-10",
        "李四": "2026-04-08",
        "王五": "2026-04-01",
    }
    # 默认返回昨天的日期（演示）
    default_date = (date.today() - timedelta(days=1)).isoformat()
    return mock_last_deposit.get(user_name, default_date)
