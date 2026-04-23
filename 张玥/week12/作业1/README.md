# 作业 1：基于 chinook.db 的 NL2SQL 问答 Agent

## 题目要求

实现一个基于 `chinook.db` 的 NL2SQL 问答 Agent，能够稳定回答以下问题：

1. 数据库中总共有多少张表
2. 员工表中有多少条记录
3. 在数据库中所有客户个数和员工个数分别是多少

## 数据库路径

本作业直接复用课程中的数据库文件：

```text
Week12/04_SQL-Code-Agent-Demo/chinook.db
```

## 运行命令

在 `study` 根目录执行：

```powershell
D:\AI_study_env\miniconda3\envs\py312\python.exe .\Week12\homework\作业1\main.py
```

## 结果口径说明

“数据库中总共有多少张表”沿用 notebook 中的统计口径：

```sql
SELECT COUNT(*) FROM sqlite_master WHERE type='table';
```

因此结果是 `13` 张表，包含 SQLite 系统表：

- `sqlite_sequence`
- `sqlite_stat1`

另外两个问题的标准答案是：

- 员工表记录数：`8`
- 客户数 / 员工数：`59 / 8`

## 脚本特点

- 先读取 schema，再交给大模型生成 SQL
- 只允许执行 `SELECT` / `WITH` 查询
- 对 3 个固定问题做结果校验，避免错误 SQL 悄悄通过
- 输出包含：
  - 原始问题
  - 生成 SQL
  - 原始查询结果
  - 最终自然语言回答

