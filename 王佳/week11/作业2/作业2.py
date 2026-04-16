"""
作业2: 4-项目案例-企业职能助手，增加3个自定义的tool 工具，实现自定义的功能，并在对话框完成调用（自然语言 -》 工具选择 -》 工具执行结果）
"""

import random
from typing import Annotated
from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


@mcp.tool
def send_work_email(
        recipient: Annotated[str, "收件人邮箱地址或姓名"],
        subject: Annotated[str, "邮件主题"],
        content: Annotated[str, "邮件正文内容"],
        priority: Annotated[str, "邮件优先级：'高'、'普通'、'低'，默认为'普通'"] = "普通"
):
    """发送工作邮件到指定收件人，支持设置优先级。"""
    priority_map = {"高": "🔴", "普通": "🟡", "低": "🟢"}
    priority_icon = priority_map.get(priority, "🟡")

    email_record = {
        "status": "已发送",
        "recipient": recipient,
        "subject": subject,
        "priority": priority,
        "priority_icon": priority_icon,
        "send_time": "2026-04-16 14:30:00",
        "message_id": f"MSG-{random.randint(10000, 99999)}"
    }

    return email_record


@mcp.tool
def manage_task(
        action: Annotated[str, "操作类型：'create'(创建)、'query'(查询)、'update'(更新)、'complete'(完成)"],
        task_title: Annotated[str, "任务标题（创建时必填）"] = "",
        assignee: Annotated[str, "负责人姓名（创建时必填）"] = "",
        deadline: Annotated[str, "截止日期，格式为YYYY-MM-DD"] = "",
        task_id: Annotated[str, "任务ID（查询/更新/完成时使用）"] = ""
):
    """管理工作任务，支持创建、查询、更新和完成任务。"""
    if action == "create":
        new_task_id = f"TASK-{random.randint(1000, 9999)}"
        return {
            "action": "创建成功",
            "task_id": new_task_id,
            "title": task_title,
            "assignee": assignee,
            "deadline": deadline,
            "status": "进行中",
            "created_at": "2026-04-16 10:00:00"
        }
    elif action == "query":
        sample_tasks = [
            {
                "task_id": "TASK-1001",
                "title": "完成Q1业绩报告",
                "assignee": "李四",
                "deadline": "2026-04-20",
                "status": "进行中",
                "progress": "60%"
            },
            {
                "task_id": "TASK-1002",
                "title": "组织团队建设活动",
                "assignee": "王五",
                "deadline": "2026-04-25",
                "status": "待开始",
                "progress": "0%"
            },
            {
                "task_id": "TASK-1003",
                "title": "客户拜访计划制定",
                "assignee": "赵六",
                "deadline": "2026-04-18",
                "status": "已完成",
                "progress": "100%"
            }
        ]
        return {"total": len(sample_tasks), "tasks": sample_tasks}
    elif action == "complete":
        return {
            "action": "完成成功",
            "task_id": task_id,
            "status": "已完成",
            "completed_at": "2026-04-16 16:45:00"
        }
    else:
        return {"error": "不支持的操作类型"}


@mcp.tool
def request_office_supplies(
        employee_name: Annotated[str, "申请人姓名"],
        department: Annotated[str, "所属部门"],
        items: Annotated[str, "申领物品清单，多个物品用逗号分隔，如'笔记本x2,签字笔x5'"],
        reason: Annotated[str, "申领原因"] = "日常工作需要"
):
    """提交办公用品申领申请，包括文具、耗材等办公物资。"""
    item_list = [item.strip() for item in items.split(",")]

    request_id = f"REQ-{random.randint(10000, 99999)}"

    approval_status = random.choice(["已批准", "待审批", "待审批"])

    return {
        "request_id": request_id,
        "employee": employee_name,
        "department": department,
        "items": item_list,
        "reason": reason,
        "status": approval_status,
        "submit_time": "2026-04-16 19:30:00",
        "estimated_delivery": "2026-04-17 10:00:00",
        "note": "审批通过后将由行政部统一发放"
    }
