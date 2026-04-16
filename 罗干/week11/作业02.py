import re
from typing import Annotated, Union
import requests
import math
from datetime import datetime, timedelta
import httpx
import random
from duckduckgo_search import DDGS
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
    """Query user salary baed on the username."""

    # TODO 基于用户名，在数据库中查询，返回数据库查询结果

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000

# --- 工具 1: 数学直角三角形写笔记计算 ---
@mcp.tool()
def calculate_hypotenuse(a: float, b: float) -> float:
    """
    根据勾股定理，计算直角三角形的斜边长度。

    Args:
        a: 直角边 a 的长度
        b: 直角边 b 的长度

    Returns:
        斜边 c 的长度
    """
    if a <= 0 or b <= 0:
        raise ValueError("边长必须大于 0")

    c = math.sqrt(a ** 2 + b ** 2)
    return round(c, 4)


# ---简单的城市名称转码字典---
CITY_MAPPING = {
    "北京": "BJP", "上海": "SHH", "广州": "GZQ", "深圳": "SZQ",
    "杭州": "HZH", "南京": "NJH", "武汉": "WHN", "成都": "CDW",
    "重庆": "CQW", "西安": "XAY", "天津": "TJP", "苏州": "SZH"
}


# --- 工具 2: 查询火车票工具 ---
@mcp.tool()
def query_train_tickets(from_station: str, to_station: str, date: str = "今天") -> str:
    """
    查询中国铁路（12306）的余票信息。

    Args:
        from_station: 出发城市名称 (例如: "北京", "上海")
        to_station: 到达城市名称 (例如: "广州", "深圳")
        date: 查询日期，支持 "今天", "明天", 或 "YYYY-MM-DD" 格式

    Returns:
        包含车次、出发/到达时间、历时和余票信息的文本报告
    """

    # 1. 处理日期
    query_date = ""
    today = datetime.now().date()

    if date == "今天":
        query_date = today.strftime("%Y-%m-%d")
    elif date == "明天":
        query_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif date == "后天":
        query_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    else:
        # 尝试解析标准格式
        try:
            datetime.strptime(date, "%Y-%m-%d")
            query_date = date
        except ValueError:
            return f"❌ 日期格式错误：'{date}'，请使用 '今天', '明天' 或 'YYYY-MM-DD'。"

    # 2. 处理城市名称 (转码)
    from_code = CITY_MAPPING.get(from_station)
    to_code = CITY_MAPPING.get(to_station)

    if not from_code:
        return f"❌ 未知的出发城市：'{from_station}'。目前仅支持: {', '.join(CITY_MAPPING.keys())}"
    if not to_code:
        return f"❌ 未知的到达城市：'{to_station}'。目前仅支持: {', '.join(CITY_MAPPING.keys())}"

    # 3. 构造 12306 查询 URL
    # 使用官方接口，参数含义：leftTicket=0, date=日期, from=出发站码, to=到达站码
    url = 'https://kyfw.12306.cn/otn/leftTicket/queryG?leftTicketDTO.train_date={}&leftTicketDTO.from_station={}&leftTicketDTO.to_station={}&purpose_codes=ADULT'.format(
        query_date, from_code, to_code)
    print("######查询的条件为：",query_date, from_code, to_code)
    print("###url为", url)
    try:
        # 4. 发送请求
        # 必须设置 header,cookies
        headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "If-Modified-Since": "0",
            "Pragma": "no-cache",
            "Referer": "https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": "\"Chromium\";v=\"128\", \"Not;A=Brand\";v=\"24\", \"Google Chrome\";v=\"128\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\""
        }
        cookies = {
            "_uab_collina": "",
            "JSESSIONID": "",
            "BIGipServerotn": "",
            "BIGipServerpassport": "",
            "guidesStatus": "",
            "highContrastMode": "",
            "cursorStatus": "",
            "route": "",
            "_jc_save_fromStation": "",
            "_jc_save_toStation": "",
            "_jc_save_fromDate": "",
            "_jc_save_toDate": "",
            "_jc_save_wfdc_flag": ""
        }
        with httpx.Client() as client:
            response = client.get(url, headers=headers, cookies=cookies,timeout=10.0)

            if response.status_code != 200:
                return f"❌ 查询失败，服务器返回状态码: {response.status_code}"

            data = response.json()

            # 5. 解析结果
            if data.get("status") and data.get("data") and data["data"].get("result"):
                results = data["data"]["result"]
                # 12306 返回的数据是以 "|" 分隔的字符串，我们需要提取特定字段
                # 字段索引参考：3(车次), 8(出发时间), 9(到达时间), 10(历时), 21(二等座)
                report = f"🚄 **{from_station} -> {to_station} ({query_date})** 余票查询结果：\n\n"
                count = 0

                for item in results:
                    parts = item.split("|")
                    # 过滤掉一些无效数据
                    if len(parts) > 21:
                        train_no = parts[3]
                        start_time = parts[8]
                        arrive_time = parts[9]
                        duration = parts[10]
                        second_class = parts[21]  # 二等座余票情况 (如 "有", "无", 数字)

                        # 简单过滤，只显示有票或显示状态的
                        report += f"- **{train_no}**: {start_time}出发 -> {arrive_time}到达 (历时{duration}) | 二等座: {second_class}\n"
                        count += 1

                if count == 0:
                    report += "暂无车次信息。"

                return report
            else:
                return "ℹ️ 未查询到车次信息，可能是日期过远或线路不存在。"

    except httpx.RequestError as e:
        return f"❌ 网络请求失败: {str(e)}"


# --- 内置精简诗库 (模拟本地数据库) ---
# 为了演示效果，这里预存了几位大诗人的代表作
POETRY_DB = {
    "李白": [
        {"title": "静夜思", "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
        {"title": "将进酒", "content": "君不见，黄河之水天上来，奔流到海不复回。...天生我材必有用，千金散尽还复来。"},
        {"title": "早发白帝城", "content": "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。"}
    ],
    "杜甫": [
        {"title": "春望", "content": "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。"},
        {"title": "登高", "content": "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。"}
    ],
    "苏轼": [
        {"title": "水调歌头", "content": "明月几时有？把酒问青天。...但愿人长久，千里共婵娟。"},
        {"title": "定风波", "content": "莫听穿林打叶声，何妨吟啸且徐行。竹杖芒鞋轻胜马，谁怕？一蓑烟雨任平生。"}
    ],
    "李清照": [
        {"title": "如梦令", "content": "昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。"},
        {"title": "声声慢", "content": "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。...这次第，怎一个愁字了得！"}
    ]
}

# --- 工具3: 诗词查询 ---
@mcp.tool()
def get_poem_by_author(author: str, mood: str = "random") -> str:
    """
    根据诗人名字返回一首该作者的诗词。
    内置了李白、杜甫、苏轼、李清照等常见诗人。如果是其他诗人，会尝试联网搜索。

    Args:
        author: 诗人名字 (例如: "李白", "杜甫")
        mood: 筛选心情 (可选: "random" 随机, "happy" 豪放/快乐, "sad" 婉约/悲伤)。默认为 "random"

    Returns:
        包含诗名和内容的格式化文本
    """

    # 1. 优先从本地数据库查找
    if author in POETRY_DB:
        poems = POETRY_DB[author]
        selected_poem = None
        if mood == "random":
            selected_poem = random.choice(poems)
        else:
            # 没有复杂的标签系统，这里简单随机标签
            selected_poem = random.choice(poems)

        return f"📜 **{author} · {selected_poem['title']}**\n\n{selected_poem['content']}"

    # --- 第二步：联网搜索 (如果本地没有) ---
    try:
        # 构造搜索关键词
        query = f"{author} 最著名的诗词"

        with DDGS() as ddgs:
            # 搜索前 3 条结果
            results = list(ddgs.text(query, max_results=3))

            if not results:
                return f"🤔 在网上也没找到关于 **{author}** 的信息。"

            # 格式化搜索结果
            response_text = f"🌐 **联网搜索：{author} 相关诗词**\n\n"
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                body = result.get("body", "无摘要")
                link = result.get("href", "")

                response_text += f"**{i}. {title}**\n{body}\n[来源]({link})\n\n"

            return response_text

    except Exception as e:
        return f"❌ 联网搜索失败: {str(e)}"
