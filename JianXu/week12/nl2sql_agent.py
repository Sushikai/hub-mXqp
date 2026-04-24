import os
import sys
import sqlite3
import traceback
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = "Week12/04_SQL-Code-Agent-Demo/chinook.db"

# 替换为你自己的 DashScope API Key，或通过环境变量 OPENAI_API_KEY 传入
API_KEY = os.environ.get("OPENAI_API_KEY", "---")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = "qwen-max"

# ────────────────────────────────────────────────
# 工具：获取数据库 schema（表名 + 字段）
# ────────────────────────────────────────────────

def get_schema(db_path: str) -> str:
    """读取所有表的建表语句，拼成给 LLM 的 schema 上下文。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchone()  # 先测试一下能不能读
        cursor.execute(f"PRAGMA table_info({table});")
        cols = cursor.fetchall()
        col_defs = ", ".join(f"{col[1]} {col[2]}" for col in cols)
        schema_parts.append(f"表 {table}({col_defs})")

    conn.close()
    return "\n".join(schema_parts)


def execute_sql(db_path: str, sql: str):
    """执行 SQL，返回 (列名列表, 结果行列表)。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    conn.close()
    return columns, rows


# ────────────────────────────────────────────────
# LLM 调用
# ────────────────────────────────────────────────

def call_llm(messages: list) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ────────────────────────────────────────────────
# NL2SQL Agent
# ────────────────────────────────────────────────

SQL_GEN_SYSTEM = """你是一个专业的 SQLite 数据库专家。
用户会用自然语言提问，你需要根据下面的数据库 schema 生成一条正确的 SQLite SQL 语句来回答问题。
只输出 SQL，不要有任何解释或 markdown 代码块。

数据库 schema：
{schema}
"""

ANSWER_SYSTEM = """你是一个数据分析助手，请根据用户的提问、执行的 SQL 和查询结果，
用简洁的中文自然语言给出最终答案。"""


class NL2SQLAgent:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = get_schema(db_path)
        print("=== 数据库 Schema 加载完成 ===")
        print(self.schema)
        print()

    def ask(self, question: str) -> str:
        print(f"【提问】{question}")

        # Step 1: 生成 SQL
        sql = call_llm([
            {"role": "system", "content": SQL_GEN_SYSTEM.format(schema=self.schema)},
            {"role": "user", "content": question},
        ])
        # 去掉可能残留的 markdown 代码块标记
        sql = sql.strip().strip("```sql").strip("```").strip()
        print(f"【生成 SQL】{sql}")

        # Step 2: 执行 SQL
        try:
            columns, rows = execute_sql(self.db_path, sql)
        except Exception:
            err = traceback.format_exc()
            print(f"【SQL 执行失败】{err}")
            return f"SQL 执行失败：{err}"

        result_str = f"列名：{columns}\n结果：{rows}"
        print(f"【执行结果】{result_str}")

        # Step 3: 自然语言回答
        answer = call_llm([
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"提问：{question}\nSQL：{sql}\n查询结果：{result_str}"},
        ])
        print(f"【最终回答】{answer}")
        print()
        return answer


# ────────────────────────────────────────────────
# 主程序：回答三个问题
# ────────────────────────────────────────────────

if __name__ == "__main__":
    agent = NL2SQLAgent(DB_PATH)

    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少",
    ]

    for q in questions:
        agent.ask(q)
        print("-" * 60)
