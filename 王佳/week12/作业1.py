"""
参考sql agent，实现一下基于 chinook.db 数据集进行问答agent（nl2sql），需要能回答如下提问：
• 提问1: 数据库中总共有多少张表；
• 提问2: 员工表中有多少条记录
• 提问3: 在数据库中所有客户个数和员工个数分别是多少
"""
import sqlite3
from typing import List, Dict, Tuple, Optional
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text
import re

class ChinookDBParser:
    """Chinook 数据库解析器"""

    def __init__(self, db_path: str = './04_SQL-Code-Agent-Demo/chinook.db'):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.conn = self.engine.connect()
        self.inspector = inspect(self.engine)

        # 获取所有表名(排除系统表)
        self.table_names = [
            name for name in self.inspector.get_table_names()
            if not name.startswith('sqlite_')
        ]

        # 缓存表信息
        self._table_info = {}
        for table_name in self.table_names:
            columns = self.inspector.get_columns(table_name)
            self._table_info[table_name] = {
                'columns': [col['name'] for col in columns],
                'column_details': columns
            }

        print(f"数据库加载成功！共 {len(self.table_names)} 张表")

    def get_table_count(self) -> int:
        """获取表总数"""
        return len(self.table_names)

    def get_table_row_count(self, table_name: str) -> int:
        """获取指定表的记录数"""
        if table_name not in self.table_names:
            raise ValueError(f"表 '{table_name}' 不存在")

        table = Table(table_name, MetaData(), autoload_with=self.engine)
        result = self.conn.execute(select(func.count()).select_from(table)).fetchone()[0]
        return result

    def execute_sql(self, sql: str) -> List[Tuple]:
        """ 执行SQL 查询"""
        try:
            result = self.conn.execute(text(sql)).fetchall()
            return result
        except Exception as e:
            raise Exception(f"SQL 执行错误：{e}")

    def close(self):
        """关闭连接"""
        self.conn.close()


class NL2SQLConverter:
    """自然语言到 SQL 转换器"""

    def __init__(self):
        # 中文表名映射
        self.table_mapping = {
            '员工表': 'employees',
            '员工': 'employees',
            '客户表': 'customers',
            '客户': 'customers',
            '订单表': 'invoices',
            '曲目表': 'tracks',
            '专辑表': 'albums',
            '艺术家表': 'artists',
        }

    def convert(self, question: str) -> str:
        """将自然语言转换为 SQL"""
        question_lower = question.lower().strip()

        # 模式1: 数据库表总数
        if any(kw in question_lower for kw in ['多少张表', '几张表']) and '数据库' in question_lower:
            return "SELECT COUNT(*) AS table_count FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"

        # 模式2: 多表记录数对比
        if '分别' in question_lower or '各是' in question_lower:
            tables = self._extract_tables(question_lower)
            if len(tables) >= 2:
                queries = []
                for table_name in tables:
                    queries.append(f"SELECT '{table_name}' AS table_name, COUNT(*) AS record_count FROM {table_name}")
                return " UNION ALL ".join(queries)

        # 模式3: 单表记录数
        for chinese_name, table_name in self.table_mapping.items():
            if chinese_name in question_lower:
                if any(kw in question_lower for kw in ['多少', '几条', '个数', '数量']):
                    return f"SELECT COUNT(*) AS record_count FROM {table_name}"

        raise ValueError(f"无法理解的问题: {question}")

    def _extract_tables(self, question: str) -> List[str]:
        """从问题中提取表名"""
        found_tables = []
        for chinese_name, table_name in self.table_mapping.items():
            if chinese_name in question:
                if table_name not in found_tables:
                    found_tables.append(table_name)
        return found_tables


class ChatBIAgent:
    """ChatBI 问答 Agent"""

    def __init__(self, db_path):
        self.parser = ChinookDBParser(db_path)
        self.converter = NL2SQLConverter()
        print("ChatBI Agent 初始化完成！\n")

    def ask(self, question: str) -> Dict:
        """处理用户提问"""
        print(f"{'=' * 60}")
        print(f"问题：{question}")
        print(f"{'=' * 60}")

        try:
            # 转换为 SQL
            sql = self.converter.convert(question)
            print(f"SQL: {sql}")

            # 执行查询
            result = self.parser.execute_sql(sql)
            print(f"结果：{result}")

            # 格式化答案
            answer = self._format_answer(question, sql, result)
            print(f"答案: {answer}\n")

            return {
                'question': question,
                'sql': sql,
                'result': result,
                'answer': answer
            }
        except Exception as e:
            error_msg = f"错误: {str(e)}"
            print(f"{error_msg}\n")
            return {'question': question, 'answer': error_msg}

    def _format_answer(self, question: str, sql: str, result: List[Tuple]) -> str:
        """格式化答案为自然语言"""

        # 表总数
        if 'sqlite_master' in sql:
            count = result[0][0] if result else 0
            return f"数据库中总共有 {count} 张表。"

        # 单表记录数
        elif 'UNION' not in sql:
            count = result[0][0] if result else 0
            table_match = re.search(r'FROM (\w+)', sql)
            table_name = table_match.group(1) if table_match else ''

            table_cn = {
                'employees': '员工表',
                'customers': '客户表',
                'invoices': '订单表',
                'tracks': '曲目表',
                'albums': '专辑表',
                'artists': '艺术家表',
            }.get(table_name, table_name)

            return f"{table_name}中有 {count} 条记录。"

        # 多表记录数
        elif 'UNION ALL' in sql:
            parts = []
            for row in result:
                table_name = row[0]
                count = row[1]
                table_cn = {
                    'employees': '员工',
                    'customers': '客户',
                    'invoices': '订单',
                    'tracks': '曲目',
                    'albums': '专辑',
                    'artists': '艺术家',
                }.get(table_name, table_name)
                parts.append(f"{table_cn}个数为 {count}")
            return "，".join(parts) + "。"

        return f"查询结果: {result}"

    def close(self):
        """关闭连接"""
        self.parser.close()


def main():
    """主函数 - 测试三个问题 """
    # 初始化 Agent
    agent = ChatBIAgent('./04_SQL-Code-Agent-Demo/chinook.db')

    # 定义测试问题
    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？"
    ]

    # 依次回答问题
    results = []
    for question in questions:
        result = agent.ask(question)
        results.append(result)

    # 汇总输出
    print(f"\n{'=' * 60}")
    print("汇总结果")
    print(f"\n{'=' * 60}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['question']}")
        print(f"   → {result['answer']}\n")

    # 关闭连接
    agent.close()


if __name__ == '__main__':
    main()

