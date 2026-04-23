"""
NL2SQL Agent 交互式版本
支持用户输入自然语言问题进行数据库查询
"""

from nl2sql_agent import NL2SQLAgent


def interactive_mode():
    """交互式问答模式"""
    print("=" * 60)
    print("欢迎使用 Chinook 数据库 NL2SQL 问答系统！")
    print("=" * 60)
    print("\n我可以回答以下类型的问题：")
    print("  1. 数据库中总共有多少张表？")
    print("  2. 某个表中有多少条记录？（如：员工表、客户表等）")
    print("  3. 多个表的记录数分别是多少？")
    print("\n输入 'quit' 或 'exit' 退出程序\n")
    
    # 创建 Agent
    agent = NL2SQLAgent('chinook.db')
    
    try:
        while True:
            # 获取用户输入
            question = input("请输入您的问题: ").strip()
            
            # 检查退出命令
            if question.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            if not question:
                continue
            
            # 回答问题
            answer = agent.ask(question)
            print(f"\n✓ 答案: {answer}\n")
            print("-" * 60)
    
    except KeyboardInterrupt:
        print("\n\n感谢使用，再见！")
    
    finally:
        agent.close()


def demo_mode():
    """演示模式 - 展示预定义问题的答案"""
    print("=" * 60)
    print("Chinook 数据库 NL2SQL 问答系统 - 演示模式")
    print("=" * 60)
    
    # 创建 Agent
    agent = NL2SQLAgent('chinook.db')
    
    # 预定义问题列表
    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？",
        "客户表中有多少条记录？",
        "订单表中有多少条记录？",
    ]
    
    # 依次回答每个问题
    for i, question in enumerate(questions, 1):
        print(f"\n示例问题 {i}: {question}")
        answer = agent.ask(question)
        print(f"\n✓ 答案: {answer}\n")
        print("-" * 60)
    
    # 关闭连接
    agent.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_mode()
    else:
        interactive_mode()
