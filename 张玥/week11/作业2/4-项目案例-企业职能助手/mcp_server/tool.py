import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


def _calculate_base_leave_days(years_of_service: int) -> int:
    """根据司龄计算基础年假天数。"""

    if years_of_service <= 1:
        return 5
    if years_of_service <= 3:
        return 7
    if years_of_service <= 5:
        return 10
    return 15


def _is_senior_level(employee_level: str) -> bool:
    """判断员工级别是否属于高职级，演示规则仅用于作业示例。"""

    normalized_level = employee_level.strip().upper()
    return normalized_level.startswith("P7") or normalized_level.startswith("P8") or normalized_level.startswith("M1") or normalized_level.startswith("M2")


def _distribute_agenda_minutes(duration_minutes: int) -> tuple[int, int, int, int]:
    """将会议总时长按固定比例拆分成 4 个阶段，并保证总时长不丢失。"""

    opening_minutes = round(duration_minutes * 0.10)
    core_minutes = round(duration_minutes * 0.60)
    action_minutes = round(duration_minutes * 0.20)
    summary_minutes = duration_minutes - opening_minutes - core_minutes - action_minutes
    return opening_minutes, core_minutes, action_minutes, summary_minutes

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
    """Query user salary baed on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


@mcp.tool
def calculate_remaining_annual_leave(
    employee_level: Annotated[str, "员工职级，例如 P6、P7、M1"],
    years_of_service: Annotated[int, "员工司龄，按年计算"],
    used_days: Annotated[float, "当前年度已经使用的年假天数"],
):
    """Calculates the simulated remaining annual leave based on level, tenure and used days."""

    base_days = _calculate_base_leave_days(years_of_service)
    bonus_days = 2 if _is_senior_level(employee_level) else 0
    total_days = float(base_days + bonus_days)
    remaining_days = max(total_days - float(used_days), 0.0)

    return {
        "employee_level": employee_level.strip().upper(),
        "years_of_service": years_of_service,
        "used_days": float(used_days),
        "base_days": base_days,
        "bonus_days": bonus_days,
        "total_days": total_days,
        "remaining_days": remaining_days,
        "rule_note": "演示规则：司龄 1 年及以内 5 天，3 年及以内 7 天，5 年及以内 10 天，5 年以上 15 天；P7/M1 及以上额外 +2 天。",
    }


@mcp.tool
def calculate_overtime_meal_allowance(
    overtime_hours: Annotated[float, "加班时长，单位为小时"],
    is_weekend: Annotated[bool, "是否为周末加班，True 表示周末，False 表示工作日"],
):
    """Calculates the simulated overtime meal allowance for weekdays or weekends."""

    if overtime_hours < 2:
        allowance = 0
        rule = "加班不足 2 小时，无餐补。"
    elif overtime_hours < 4:
        allowance = 30 if is_weekend else 20
        rule = "加班 2 到 4 小时，工作日 20 元，周末 30 元。"
    else:
        allowance = 60 if is_weekend else 40
        rule = "加班 4 小时及以上，工作日 40 元，周末 60 元。"

    return {
        "overtime_hours": float(overtime_hours),
        "is_weekend": bool(is_weekend),
        "allowance": allowance,
        "rule_note": rule,
    }


@mcp.tool
def generate_meeting_agenda(
    meeting_topic: Annotated[str, "会议主题"],
    attendee_count: Annotated[int, "参会人数"],
    duration_minutes: Annotated[int, "会议总时长，单位为分钟"],
):
    """Generates a simulated meeting agenda based on topic, attendee count and duration."""

    opening_minutes, core_minutes, action_minutes, summary_minutes = _distribute_agenda_minutes(duration_minutes)

    return {
        "meeting_topic": meeting_topic,
        "attendee_count": attendee_count,
        "duration_minutes": duration_minutes,
        "agenda_items": [
            {
                "stage": "开场与目标同步",
                "duration_minutes": opening_minutes,
                "suggestion": "主持人说明会议背景、目标和预期输出。",
            },
            {
                "stage": "核心议题讨论",
                "duration_minutes": core_minutes,
                "suggestion": "围绕主题展开重点讨论，确保关键问题充分交流。",
            },
            {
                "stage": "行动项确认",
                "duration_minutes": action_minutes,
                "suggestion": "明确负责人与截止时间，沉淀待办事项。",
            },
            {
                "stage": "总结与收尾",
                "duration_minutes": summary_minutes,
                "suggestion": "快速回顾结论，确认后续协作方式。",
            },
        ],
        "rule_note": "演示规则：会议时长按 10% 开场、60% 核心讨论、20% 行动项、10% 总结分配。",
    }
