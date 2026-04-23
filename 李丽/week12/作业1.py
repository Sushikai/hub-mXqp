import sqlite3
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def ask_glm(messages, nretry=5):
    if nretry == 0:
        return None

    try:
        completion = client.chat.completions.create(
            model="glm-4-flashx",
            messages=messages,
            temperature=0.1
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Exception: {e}")
        time.sleep(1)
        return ask_glm(messages, nretry-1)

DB_PATH = "chinook.db"

class NL2SQLAgent:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = self._get_db_schema()

    def _get_db_schema(self) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_info = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
            schema_info.append(f"Table '{table_name}' has columns: {col_info}")
            
        conn.close()
        return "\n".join(schema_info)

    def execute_sql(self, sql: str):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return result
        except Exception as e:
            return f"Error executing SQL: {e}"

    def ask(self, question: str):
        print(f"提问: {question}")
        
        prompt = f"""You are a database expert. I have a SQLite database with the following schema:
{self.schema}

Based on the schema, please write a SQL query to answer the following question:
Question: {question}

Return ONLY the SQL query, without any markdown formatting or explanation. Ensure the SQL query is valid for SQLite.
If multiple counts are requested, return them in a single SELECT statement.
"""
        sql_query = ask_glm([{"role": "user", "content": prompt}])
        if not sql_query:
            return "Failed to generate SQL."
            
        sql_query = sql_query.strip().replace("```sql", "").replace("```", "").strip()
        print(f"生成SQL: {sql_query}")
        
        result = self.execute_sql(sql_query)
        print(f"执行结果: {result}")
        
        format_prompt = f"""你是一个智能问答助手。根据用户的提问、执行的SQL查询以及数据库返回的结果，用自然语言简短直接地回答用户的问题。
提问: {question}
SQL: {sql_query}
结果: {result}

请直接用中文给出答案：
"""
        answer = ask_glm([{"role": "user", "content": format_prompt}])
        print(f"Agent回答: {answer}\n")
        return answer

if __name__ == "__main__":
    agent = NL2SQLAgent(DB_PATH)
    
    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]
    
    for q in questions:
        agent.ask(q)
