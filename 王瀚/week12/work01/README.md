# NL2SQL Agent - Chinook 数据库问答系统

基于 Chinook 数据库的自然语言到 SQL 转换（NL2SQL）问答系统，使用百炼 API（DashScope）实现智能转换，能够将用户的自然语言问题转换为 SQL 查询并返回答案。

## 功能特性

✅ 使用百炼 API（通义千问）实现智能 NL2SQL 转换  
✅ 支持自然语言提问，自动转换为 SQL 查询  
✅ 自动生成自然语言答案  
✅ 支持交互式问答和演示模式  
✅ 完整的数据库 schema 信息提供给 AI  

## 支持的问题类型

### 1. 查询数据库表总数
- **示例**: "数据库中总共有多少张表？"
- **答案**: 数据库中总共有 11 张表。

### 2. 查询单个表的记录数
- **示例**: "员工表中有多少条记录？"
- **答案**: 员工表中共有 8 条记录。

支持的表：
- 员工表 (employees)
- 客户表 (customers)
- 订单表 (invoices)
- 曲目表 (tracks)
- 专辑表 (albums)
- 艺术家表 (artists)
- 流派表 (genres)
- 播放列表 (playlists)

### 3. 查询多个表的记录数
- **示例**: "在数据库中所有客户个数和员工个数分别是多少？"
- **答案**: 数据库中客户个数为 59，员工个数为 8。

## 快速开始

### 1. 设置环境变量

在使用之前，需要设置百炼 API Key 到环境变量：

**Windows (PowerShell):**
```powershell
$env:DASHSCOPE_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set DASHSCOPE_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 2. 安装依赖

```bash
pip install openai
```

### 3. 运行程序

**方式一：使用快速启动脚本（推荐）**

Windows PowerShell:
```powershell
.\run.ps1
```

脚本会自动检查环境变量并提供三种运行模式选择。

**方式二：直接运行**

```bash
python nl2sql_interactive.py --demo
```

这将展示系统如何回答预定义的问题。

### 4. 运行交互式模式

```bash
python nl2sql_interactive.py
```

然后输入您的自然语言问题，系统会自动回答。

### 5. 直接运行测试脚本

```bash
python nl2sql_agent.py
```

这将运行预设的三个问题并显示答案。

## 项目结构

```
04_SQL-Code-Agent-Demo/
├── chinook.db              # Chinook 示例数据库
├── nl2sql_agent.py         # NL2SQL Agent 核心实现
├── nl2sql_interactive.py   # 交互式界面
├── check_db.py            # 数据库检查脚本
├── sql-agent.ipynb        # 原始 SQL Agent 笔记本
├── code-agent.ipynb       # Code Agent 笔记本
└── README.md              # 项目说明文档
```

## 技术实现

### 核心组件

1. **数据库 Schema 提取** (`_get_database_schema`)
   - 自动获取所有表名、字段信息和记录数
   - 为 AI 提供完整的数据库结构信息

2. **NL2SQL 转换** (`nl2sql`)
   - 使用百炼 API（通义千问）将自然语言转换为 SQL
   - 提供数据库 schema 作为上下文
   - 支持复杂的 SQL 查询生成

3. **SQL 执行** (`execute_sql`)
   - 使用 SQLite 执行生成的 SQL 查询
   - 返回查询结果或错误信息

4. **答案生成** (`generate_answer`)
   - 使用百炼 AI 将查询结果转换为自然语言
   - 生成友好、易懂的答案

### 工作流程

```
用户问题 → 提供数据库schema → 百炼AI生成SQL → 执行SQL → 百炼AI生成答案 → 返回答案
```

### 使用的模型

- **模型**: qwen-max（通义千问）
- **API**: 百炼 DashScope API
- **特点**: 强大的中文理解能力，优秀的 SQL 生成能力

## 测试用例

### 提问 1: 数据库中总共有多少张表？
```
生成的 SQL: SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
查询结果: [(11,)]
答案: 数据库中共有11张表。
```

### 提问 2: 员工表中有多少条记录？
```
生成的 SQL: SELECT COUNT(*) FROM employees
查询结果: [(8,)]
答案: 员工表中共有8条记录。
```

### 提问 3: 在数据库中所有客户个数和员工个数分别是多少？
```
生成的 SQL: SELECT (SELECT COUNT(*) FROM customers) AS customer_count, (SELECT COUNT(*) FROM employees) AS employee_count
查询结果: [(59, 8)]
答案: 数据库中共有59位客户和8位员工。
```

## 扩展建议

如果需要支持更多复杂查询，可以：

1. **增强提示词**：在提示词中添加更多示例和说明
2. **使用更强大的模型**：如 qwen-plus 或 qwen-turbo
3. **添加 Few-shot 示例**：在提示词中提供示例问题及其 SQL
4. **支持多轮对话**：实现上下文记忆，支持追问和澄清
5. **添加 SQL 验证**：在执行前验证 SQL 的安全性

## 依赖

- Python 3.7+
- SQLite3（Python 内置）
- openai（Python SDK）

安装依赖：
```bash
pip install openai
```

## 环境变量

- `DASHSCOPE_API_KEY`: 百炼 API Key（必需）

获取 API Key：
1. 访问 [百炼控制台](https://bailian.console.aliyun.com/)
2. 创建或获取 API Key
3. 设置到环境变量中

## 许可证

本项目用于学习和演示目的。
