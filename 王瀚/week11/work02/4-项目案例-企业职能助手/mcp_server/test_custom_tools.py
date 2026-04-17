"""
测试自定义工具的脚本
运行此脚本以验证3个新工具是否正常工作
"""
import asyncio
from custom_tools import mcp

async def test_custom_tools():
    print("=" * 60)
    print("测试自定义工具")
    print("=" * 60)
    
    # 测试1: 查询员工信息
    print("\n【测试1】查询员工信息 - 张三")
    result1 = mcp._tool_manager.get_tool_function("query_employee_info")
    if result1:
        employee_info = result1(employee_name="张三")
        print(f"结果: {employee_info}")
    else:
        print("工具未找到")
    
    # 测试2: 查询会议室可用性
    print("\n【测试2】查询会议室可用性 - 2026-04-20 09:00-10:00")
    result2 = mcp._tool_manager.get_tool_function("check_meeting_room_availability")
    if result2:
        room_info = result2(date="2026-04-20", time_slot="09:00-10:00")
        print(f"结果: {room_info}")
    else:
        print("工具未找到")
    
    # 测试3: 计算工作日
    print("\n【测试3】计算工作日 - 2026-04-01 到 2026-04-30")
    result3 = mcp._tool_manager.get_tool_function("calculate_workdays")
    if result3:
        workdays = result3(start_date="2026-04-01", end_date="2026-04-30")
        print(f"结果: {workdays}")
    else:
        print("工具未找到")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_custom_tools())
