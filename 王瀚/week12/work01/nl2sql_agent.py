"""
NL2SQL Agent - 基于 Chinook 数据库的自然语言到 SQL 转换问答系统
使用百炼 API (DashScope) 实现智能 NL2SQL 转换
"""

import sqlite3
import re
import os
from typing import List, Dict, Any, Tuple, Union
from openai import OpenAI


class NL2SQLAgent:
    """自然语言到 SQL 转换的智能体"""
    
    def __init__(self, db_path: str = 'chinook.db'):
        """初始化 Agent
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # 初始化百炼 API 客户端（使用环境变量中的 API Key）
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("未找到环境变量 DASHSCOPE_API_KEY，请设置百炼 API Key")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 获取数据库元数据
        self.tables_info = self._get_database_schema()
        
        # 系统提示词
        self.system_prompt = self._build_system_prompt()
        
        # NL2SQL 提示词
        self.nl2sql_prompt = """你是一个专业的SQL专家，擅长将自然语言问题转换为SQL查询。

数据库信息：
{schema_info}

注意：查询数据库表数量时，请排除 SQLite 系统表（表名以 'sqlite_' 开头的表）。

用户问题：{question}

请将上述自然语言问题转换为SQL查询语句。要求：
1. 只输出SQL语句，不要输出其他内容
2. SQL语句应该能够正确回答用户的问题
3. 使用标准的SQLite语法
4. 如果问题涉及计数，使用COUNT()函数
5. 如果问题需要查询特定表的信息，请使用正确的表名和字段名
6. 查询表总数时，使用条件：WHERE type='table' AND name NOT LIKE 'sqlite_%'

SQL查询："""

        # 答案生成提示词
        self.answer_prompt = """你是一个专业的数据分析师，请根据以下信息用自然语言回答问题。

原始问题：{question}
SQL查询：{sql}
查询结果：{result}

请用简洁明了的中文回答："""
    
    def _get_database_schema(self) -> Dict[str, Any]:
        """获取数据库的 schema 信息"""
        schema = {}
        
        # 获取所有表名
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in self.cursor.fetchall()]
        
        # 排除 sqlite 内部表
        tables = [t for t in tables if not t.startswith('sqlite_')]
        
        for table in tables:
            # 获取表的列信息
            self.cursor.execute(f"PRAGMA table_info({table});")
            columns = self.cursor.fetchall()
            
            schema[table] = {
                'columns': [{'name': col[1], 'type': col[2]} for col in columns]
            }
            
            # 获取表的记录数
            self.cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = self.cursor.fetchone()[0]
            schema[table]['row_count'] = count
        
        return schema
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词，包含数据库结构信息"""
        schema_text = "数据库包含以下表：\n"
        
        for table_name, info in self.tables_info.items():
            schema_text += f"\n表名: {table_name} (共 {info['row_count']} 条记录)\n"
            schema_text += "字段:\n"
            for col in info['columns']:
                schema_text += f"  - {col['name']} ({col['type']})\n"
        
        return schema_text
    
    def nl2sql(self, question: str) -> str:
        """将自然语言转换为 SQL
        
        Args:
            question: 自然语言问题
            
        Returns:
            SQL 查询语句
        """
        prompt = self.nl2sql_prompt.format(
            schema_info=self.system_prompt,
            question=question
        )
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql = response.choices[0].message.content.strip()
            
            # 清理 SQL（移除可能的 markdown 格式）
            sql = re.sub(r'^```sql\s*', '', sql)
            sql = re.sub(r'\s*```$', '', sql)
            sql = sql.strip()
            
            return sql
            
        except Exception as e:
            print(f"调用百炼 API 失败: {e}")
            return None
    
    def execute_sql(self, sql: str) -> Union[List, str]:
        """执行 SQL 查询
        
        Args:
            sql: SQL 查询语句
            
        Returns:
            查询结果或错误信息
        """
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            error_msg = f"SQL执行错误: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_answer(self, question: str, sql: str, result: List) -> str:
        """生成自然语言答案
        
        Args:
            question: 原始问题
            sql: 执行的 SQL
            result: 查询结果
            
        Returns:
            自然语言答案
        """
        result_str = str(result)
        
        prompt = self.answer_prompt.format(
            question=question,
            sql=sql,
            result=result_str
        )
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"生成答案失败: {e}")
            # 如果大模型调用失败，直接返回结果
            return f"查询结果: {result}"
    
    def ask(self, question: str) -> str:
        """问答接口
        
        Args:
            question: 自然语言问题
            
        Returns:
            自然语言答案
        """
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        # Step 1: NL2SQL 转换
        print("\n[Step 1] 使用百炼 AI 将自然语言转换为 SQL...")
        sql = self.nl2sql(question)
        print(f"生成的 SQL: {sql}")
        
        if not sql:
            return "抱歉，无法生成有效的 SQL 查询。"
        
        # Step 2: 执行 SQL
        print("\n[Step 2] 执行 SQL 查询...")
        result = self.execute_sql(sql)
        print(f"查询结果: {result}")
        
        if isinstance(result, str):  # 错误信息
            return f"查询执行失败: {result}"
        
        # Step 3: 生成自然语言答案
        print("\n[Step 3] 使用百炼 AI 生成自然语言答案...")
        answer = self.generate_answer(question, sql, result)
        print(f"最终答案: {answer}")
        
        return answer
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()


def main():
    """测试 NL2SQL Agent"""
    # 创建 Agent
    agent = NL2SQLAgent('chinook.db')
    
    # 测试问题列表
    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？"
    ]
    
    # 依次回答每个问题
    for question in questions:
        answer = agent.ask(question)
        print(f"\n✓ 答案: {answer}\n")
        print("-" * 60)
    
    # 关闭连接
    agent.close()


if __name__ == "__main__":
    main()
