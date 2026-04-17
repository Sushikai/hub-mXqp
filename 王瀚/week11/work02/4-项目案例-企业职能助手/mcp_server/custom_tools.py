from typing import Annotated
from datetime import datetime, timedelta
from fastmcp import FastMCP

mcp = FastMCP(
    name="Custom-MCP-Server",
    instructions="""This server contains custom enterprise tools for employee info, meeting rooms, and workday calculations.""",
)

# 模拟员工数据库
EMPLOYEE_DATABASE = {
    "张三": {"部门": "技术部", "职位": "高级工程师", "工号": "EMP001", "入职日期": "2020-03-15", "邮箱": "zhangsan@company.com"},
    "李四": {"部门": "市场部", "职位": "市场经理", "工号": "EMP002", "入职日期": "2019-07-20", "邮箱": "lisi@company.com"},
    "王五": {"部门": "财务部", "职位": "财务主管", "工号": "EMP003", "入职日期": "2021-01-10", "邮箱": "wangwu@company.com"},
    "赵六": {"部门": "人力资源部", "职位": "HR专员", "工号": "EMP004", "入职日期": "2022-05-08", "邮箱": "zhaoliu@company.com"},
    "钱七": {"部门": "技术部", "职位": "前端开发工程师", "工号": "EMP005", "入职日期": "2023-02-14", "邮箱": "qianqi@company.com"},
}

# 模拟会议室数据
MEETING_ROOMS = {
    "会议室A": {"容量": 10, "楼层": "3楼", "设备": ["投影仪", "白板", "视频会议系统"]},
    "会议室B": {"容量": 6, "楼层": "3楼", "设备": ["电视", "白板"]},
    "会议室C": {"容量": 20, "楼层": "5楼", "设备": ["投影仪", "音响系统", "视频会议系统", "白板"]},
    "会议室D": {"容量": 4, "楼层": "2楼", "设备": ["电视"]},
    "会议室E": {"容量": 15, "楼层": "5楼", "设备": ["投影仪", "白板", "电话会议系统"]},
}


@mcp.tool
def query_employee_info(employee_name: Annotated[str, "员工姓名，例如：'张三' 或 '李四'"]):
    """根据员工姓名查询员工的基本信息，包括部门、职位、工号、入职日期和邮箱。"""
    if employee_name in EMPLOYEE_DATABASE:
        info = EMPLOYEE_DATABASE[employee_name]
        return {
            "姓名": employee_name,
            "工号": info["工号"],
            "部门": info["部门"],
            "职位": info["职位"],
            "入职日期": info["入职日期"],
            "邮箱": info["邮箱"]
        }
    else:
        return f"未找到员工 '{employee_name}' 的信息。可用员工：{', '.join(EMPLOYEE_DATABASE.keys())}"


@mcp.tool
def check_meeting_room_availability(date: Annotated[str, "查询日期，格式：'YYYY-MM-DD'，例如：'2026-04-20'"], 
                                     time_slot: Annotated[str, "时间段，例如：'09:00-10:00' 或 '14:00-15:00'"]):
    """查询指定日期和时间段的会议室可用状态，返回所有会议室的可用性信息。"""
    try:
        query_date = datetime.strptime(date, "%Y-%m-%d")
        
        # 模拟预约数据（实际应该从数据库查询）
        # 这里简单模拟：工作日的大部分时间段都是可用的
        booked_rooms = {}
        
        if query_date.weekday() < 5:  # 工作日
            # 模拟某些时间段已被预约
            if "09:00-10:00" in time_slot:
                booked_rooms = {"会议室A": "技术部周会", "会议室C": "产品评审"}
            elif "14:00-15:00" in time_slot:
                booked_rooms = {"会议室B": "面试", "会议室E": "客户会议"}
            elif "10:00-11:00" in time_slot:
                booked_rooms = {"会议室D": "一对一沟通"}
        
        availability = []
        for room_name, room_info in MEETING_ROOMS.items():
            if room_name in booked_rooms:
                availability.append({
                    "会议室": room_name,
                    "状态": "已预约",
                    "预约事由": booked_rooms[room_name],
                    "容量": room_info["容量"],
                    "位置": room_info["楼层"],
                    "设备": ", ".join(room_info["设备"])
                })
            else:
                availability.append({
                    "会议室": room_name,
                    "状态": "可用",
                    "容量": room_info["容量"],
                    "位置": room_info["楼层"],
                    "设备": ", ".join(room_info["设备"])
                })
        
        return {
            "查询日期": date,
            "时间段": time_slot,
            "会议室状态": availability
        }
    except ValueError:
        return "日期格式错误，请使用 'YYYY-MM-DD' 格式，例如：'2026-04-20'"


@mcp.tool
def calculate_workdays(start_date: Annotated[str, "开始日期，格式：'YYYY-MM-DD'，例如：'2026-04-01'"], 
                       end_date: Annotated[str, "结束日期，格式：'YYYY-MM-DD'，例如：'2026-04-30'"]):
    """计算两个日期之间的工作日天数（不包括周末），可用于计算项目工期、休假天数等。"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start > end:
            return "错误：开始日期不能晚于结束日期"
        
        workdays = 0
        weekends = 0
        current = start
        
        while current <= end:
            if current.weekday() < 5:  # 周一到周五
                workdays += 1
            else:  # 周六和周日
                weekends += 1
            current += timedelta(days=1)
        
        return {
            "开始日期": start_date,
            "结束日期": end_date,
            "工作日天数": workdays,
            "周末天数": weekends,
            "总天数": (end - start).days + 1,
            "说明": "工作日不包括周六和周日，法定节假日需要另行扣除"
        }
    except ValueError:
        return "日期格式错误，请使用 'YYYY-MM-DD' 格式，例如：'2026-04-01'"
