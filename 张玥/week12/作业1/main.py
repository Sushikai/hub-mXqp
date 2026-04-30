"""
作业 1：基于 chinook.db 的 NL2SQL 问答 Agent。

实现目标：
1. 读取数据库 schema；
2. 让大模型根据自然语言生成结构化 SQL；
3. 仅执行只读 SQL；
4. 将查询结果再总结为自然语言答案；
5. 对指定的 3 个问题做结果一致性校验，保证演示稳定。
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_NAME = "qwen-max"
ROOT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = ROOT_DIR / "Week12" / "04_SQL-Code-Agent-Demo" / "chinook.db"


class SQLGenerationOutput(BaseModel):
    """SQL 生成 Agent 的结构化输出。"""

    question_type: Literal["table_count", "employee_count", "customer_employee_count"] = Field(
        description="当前问题所属的固定任务类型。"
    )
    sql: str = Field(description="根据自然语言问题生成的只读 SQL 语句。")
    reasoning: str = Field(description="对 SQL 生成思路的简要解释。")


class AnswerSummaryOutput(BaseModel):
    """回答总结 Agent 的结构化输出。"""

    answer: str = Field(description="面向用户的最终自然语言回答。")
    key_points: list[str] = Field(description="用于辅助解释的关键点列表。")


@dataclass(frozen=True)
class HomeworkQuestion:
    """单个作业问题配置。"""

    title: str
    question: str
    question_type: Literal["table_count", "employee_count", "customer_employee_count"]
    expected_result: list[tuple[Any, ...]]


def configure_runtime() -> str:
    """配置运行环境并返回模型名称。"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY，请先在终端中配置后再运行本脚本。")

    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    return os.getenv("HOMEWORK_WEEK12_MODEL", DEFAULT_MODEL_NAME)


def load_schema_summary(db_path: Path) -> str:
    """读取数据库 schema，并整理成给大模型理解的文本。

    先读取 schema 的原因：
    - 大模型不知道数据库里有哪些表和字段；
    - 只有提前告诉它真实 schema，它才更可能生成正确 SQL；
    - 对 NL2SQL 来说，schema 是最基本的上下文。
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        table_names = [row[0] for row in cursor.fetchall()]

        schema_lines: list[str] = []
        for table_name in table_names:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_text = ", ".join(f"{col[1]} ({col[2]})" for col in columns)
            schema_lines.append(f"- {table_name}: {column_text}")

    return "\n".join(schema_lines)


def build_agents(model_name: str, schema_summary: str) -> tuple[Agent, Agent]:
    """构建 SQL 生成 Agent 和回答总结 Agent。"""

    sql_agent = Agent(
        name="Week12SqlAgent",
        model=model_name,
        instructions=(
            "你是一个专业的 SQLite NL2SQL 生成助手。"
            "你只需要处理以下三类问题："
            "1. 查询数据库总表数；"
            "2. 查询 employees 表记录数；"
            "3. 同时查询 customers 和 employees 的记录数。"
            "你必须严格基于下面给出的数据库 schema 生成 SQL，不能虚构表名和字段名。\n\n"
            f"数据库 schema 如下：\n{schema_summary}\n\n"
            "输出要求："
            "1. 只输出结构化 JSON；"
            "2. sql 必须是只读查询；"
            "3. question_type 必须是 table_count、employee_count、customer_employee_count 三者之一；"
            "4. 如果问题属于“数据库总共有多少张表”，请按 SELECT COUNT(*) FROM sqlite_master WHERE type='table' 的口径统计。"
        ),
        output_type=SQLGenerationOutput,
    )

    answer_agent = Agent(
        name="Week12AnswerAgent",
        model=model_name,
        instructions=(
            "你是一个数据库问答总结助手。"
            "请根据 SQL 执行结果，用简洁准确的中文回答用户问题。"
            "不要编造额外数据，要明确指出关键数字。"
        ),
        output_type=AnswerSummaryOutput,
    )
    return sql_agent, answer_agent


def validate_read_only_sql(sql: str) -> None:
    """限制只读 SQL，避免误执行写操作。"""

    normalized_sql = sql.strip().lower().rstrip(";")
    if not normalized_sql.startswith(("select", "with")):
        raise ValueError(f"SQL 非只读查询，已拒绝执行：{sql}")


def validate_sql_against_question(question_type: str, sql: str) -> None:
    """对固定问题做 SQL 规则校验，避免明显跑偏。"""

    normalized_sql = " ".join(sql.strip().lower().split())

    required_fragments = {
        "table_count": ["sqlite_master", "type='table'"],
        "employee_count": ["from employees"],
        "customer_employee_count": ["customers", "employees"],
    }

    fragments = required_fragments[question_type]
    if not all(fragment in normalized_sql for fragment in fragments):
        raise ValueError(
            f"生成的 SQL 与问题类型 {question_type} 不匹配。\n"
            f"SQL: {sql}"
        )


def execute_sql(db_path: Path, sql: str) -> list[tuple[Any, ...]]:
    """执行只读 SQL。"""

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()


def validate_query_result(question_type: str, query_result: list[tuple[Any, ...]]) -> None:
    """对指定问题做答案一致性检查。"""

    expected_map = {
        "table_count": [(13,)],
        "employee_count": [(8,)],
        "customer_employee_count": [(59, 8)],
    }

    expected = expected_map[question_type]
    if query_result != expected:
        raise ValueError(
            f"SQL 执行结果与题目标准答案不一致。\n"
            f"问题类型: {question_type}\n"
            f"期望结果: {expected}\n"
            f"实际结果: {query_result}"
        )


def format_query_result(query_result: list[tuple[Any, ...]]) -> str:
    """格式化原始查询结果，方便打印和传给总结 Agent。"""

    return json.dumps(query_result, ensure_ascii=False)


def build_summary_prompt(question: str, sql: str, query_result: list[tuple[Any, ...]]) -> str:
    """构造总结 Agent 的输入。"""

    return (
        f"用户问题：{question}\n"
        f"执行 SQL：{sql}\n"
        f"查询结果：{format_query_result(query_result)}\n"
        "请给出准确、简洁的中文回答。"
    )


def run_single_question(
    question_config: HomeworkQuestion,
    sql_agent: Agent,
    answer_agent: Agent,
) -> None:
    """执行单个问题的 NL2SQL 问答流程。"""

    print("\n" + "=" * 90)
    print(f"题目：{question_config.title}")
    print(f"原始问题：{question_config.question}")

    sql_result = Runner.run_sync(sql_agent, question_config.question)
    structured_sql = sql_result.final_output_as(SQLGenerationOutput)

    print(f"识别问题类型：{structured_sql.question_type}")
    print(f"SQL 生成思路：{structured_sql.reasoning}")
    print(f"生成 SQL：\n{structured_sql.sql}")

    if structured_sql.question_type != question_config.question_type:
        raise ValueError(
            f"问题类型识别错误。期望 {question_config.question_type}，实际 {structured_sql.question_type}"
        )

    validate_read_only_sql(structured_sql.sql)
    validate_sql_against_question(structured_sql.question_type, structured_sql.sql)

    query_result = execute_sql(DB_PATH, structured_sql.sql)
    print(f"原始查询结果：{format_query_result(query_result)}")

    validate_query_result(structured_sql.question_type, query_result)
    if query_result != question_config.expected_result:
        raise ValueError(
            f"脚本内置标准答案不一致。期望 {question_config.expected_result}，实际 {query_result}"
        )

    answer_result = Runner.run_sync(
        answer_agent,
        build_summary_prompt(question_config.question, structured_sql.sql, query_result),
    )
    final_answer = answer_result.final_output_as(AnswerSummaryOutput)
    print(f"最终自然语言回答：{final_answer.answer}")
    print(f"回答要点：{json.dumps(final_answer.key_points, ensure_ascii=False)}")


def main() -> None:
    """运行 3 个固定作业问题。"""

    model_name = configure_runtime()
    schema_summary = load_schema_summary(DB_PATH)
    sql_agent, answer_agent = build_agents(model_name, schema_summary)

    questions = [
        HomeworkQuestion(
            title="提问1",
            question="数据库中总共有多少张表？",
            question_type="table_count",
            expected_result=[(13,)],
        ),
        HomeworkQuestion(
            title="提问2",
            question="员工表中有多少条记录？",
            question_type="employee_count",
            expected_result=[(8,)],
        ),
        HomeworkQuestion(
            title="提问3",
            question="在数据库中所有客户个数和员工个数分别是多少？",
            question_type="customer_employee_count",
            expected_result=[(59, 8)],
        ),
    ]

    print("开始运行 Week12 作业 1：NL2SQL 问答 Agent")
    print(f"数据库路径：{DB_PATH}")
    print(f"当前模型：{model_name}")

    for question_config in questions:
        run_single_question(question_config, sql_agent, answer_agent)

    print("\n所有固定问题运行完成。")


if __name__ == "__main__":
    main()

