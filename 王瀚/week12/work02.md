## 一、前后端分离架构

### 1.1 什么是前后端分离？

前后端分离是一种软件架构模式，将前端（用户界面）和后端（业务逻辑、数据处理）完全独立开发和部署。

### 1.2 本项目中的体现

#### 后端架构（FastAPI）

- **主入口**: [main_server.py](file:///e:/AIProject/06-stock-bi-agent/main_server.py)
  - 使用 FastAPI 框架提供 RESTful API
  - 挂载多个路由模块和用户服务
  
- **路由层** (API接口定义):
  - `routers/chat.py` - 对话相关接口
    - `POST /v1/chat/` - 发起对话（流式响应）
    - `POST /v1/chat/init` - 初始化会话
    - `POST /v1/chat/get` - 获取历史对话
    - `POST /v1/chat/delete` - 删除对话
    - `POST /v1/chat/list` - 获取对话列表
    - `POST /v1/chat/feedback` - 对话反馈
  
  - `routers/user.py` - 用户管理
  - `routers/data.py` - 数据管理
  - `routers/stock.py` - 股票相关

- **服务层** (业务逻辑):
  - `services/chat.py` - 对话管理核心逻辑
  - `services/user.py` - 用户服务
  - `services/stock.py` - 股票服务

- **数据层** (ORM模型):
  - `models/orm.py` - SQLAlchemy 数据库模型
  - `models/data_models.py` - Pydantic 数据验证模型

#### 前端交互方式

前端通过 HTTP 请求调用后端 API：
- 使用 POST 请求发送对话内容
- 接收 SSE (Server-Sent Events) 流式响应
- 通过标准化的数据模型交互（如 `RequestForChat`、`BasicResponse`）

### 1.3 前后端分离的优势

1. **独立开发**: 前端和后端可以并行开发，提高开发效率
2. **技术栈灵活**: 前端可用 React/Vue/Angular，后端用 Python/FastAPI
3. **易于扩展**: 可独立部署和扩容，支持微服务架构
4. **接口标准化**: 通过 API 契约进行数据交互，降低耦合度
5. **维护性强**: 前后端职责清晰，便于后期维护和迭代

---

## 二、历史对话存储与上下文传递机制

### 2.1 双层存储架构

#### 第一层：关系型数据库存储 (SQLite - sever.db)

**表1: ChatSessionTable** (会话元数据表)
```python
- id: 主键
- user_id: 用户ID（外键关联user表）
- session_id: 会话唯一标识（12位随机字符串）
- title: 对话标题
- start_time: 开始时间
- feedback: 反馈状态
- feedback_time: 反馈时间
```

**表2: ChatMessageTable** (消息记录表)
```python
- id: 主键
- chat_id: 关联会话ID（外键关联chat_session表）
- role: 角色类型（system/user/assistant）
- content: 消息内容（Text类型）
- generated_sql: 生成的SQL语句（可选）
- generated_code: 生成的代码（可选）
- create_time: 创建时间
- feedback: 消息反馈
- feedback_time: 反馈时间
```

#### 第二层：Agent Framework Session 存储 (SQLite - conversations.db)

使用 OpenAI Agents SDK 提供的 `AdvancedSQLiteSession`:
```python
session = AdvancedSQLiteSession(
    session_id=session_id,      # 与系统对话ID关联
    db_path="./assert/conversations.db",
    create_tables=True
)
```

**作用**: 维护对话状态和上下文，自动管理大模型所需的对话历史。

### 2.2 历史对话作为大模型输入的完整流程

#### 步骤1: 初始化会话

当用户首次发起对话时（[services/chat.py#L75-L106](file:///e:/AIProject/06-stock-bi-agent/services/chat.py#L75-L106)）:

```python
def init_chat_session(user_name, user_question, session_id, task):
    # 1. 创建会话记录
    chat_session_record = ChatSessionTable(
        user_id=user_id,
        session_id=session_id,
        title=user_question,
    )
    
    # 2. 存储 System Message（系统提示词）
    message_record = ChatMessageTable(
        chat_id=chat_session_record.id,
        role="system",
        content=get_init_message(task)  # 从Jinja2模板生成
    )
```

**System Message 模板** ([templates/chat_start_system_prompt.jinjia2](file:///e:/AIProject/06-stock-bi-agent/templates/chat_start_system_prompt.jinjia2)):
- 根据任务类型（股票分析/数据BI/普通对话）生成不同的提示词
- 包含当前时间、语言偏好等上下文信息

#### 步骤2: 用户提问时保存消息

```python
# 保存用户消息到数据库
append_message2db(session_id, "user", content)
```

#### 步骤3: 调用大模型时传递历史上下文

```python
# Agent 初始化
agent = Agent(
    name="Assistant",
    instructions=instructions,  # 系统提示词
    model=OpenAIChatCompletionsModel(
        model=os.environ["OPENAI_MODEL"],
        openai_client=external_client,
    ),
)

# 流式调用大模型（关键：session参数）
result = Runner.run_streamed(
    agent, 
    input=content,      # 当前用户输入
    session=session     # AdvancedSQLiteSession 自动包含历史对话
)
```

**关键机制**: 
- `AdvancedSQLiteSession` 对象会自动从 `conversations.db` 中加载该 `session_id` 的所有历史对话记录
- 将历史消息（system + 历史 user/assistant 消息）作为上下文传递给大模型
- 大模型基于完整上下文生成回复

#### 步骤4: 保存助手回复

```python
# 流式接收大模型输出
assistant_message = ""
async for event in result.stream_events():
    if event.type == "raw_response_event":
        if isinstance(event.data, ResponseTextDeltaEvent):
            yield f"{event.data.delta}"  # SSE 流式返回前端
            assistant_message += event.data.delta

# 保存助手回答到数据库
append_message2db(session_id, "assistant", assistant_message)
```

#### 步骤5: 获取历史对话

```python
def get_chat_sessions(session_id: str):
    # 从数据库查询该会话的所有消息
    chat_messages = session.query(ChatMessageTable) \
        .join(ChatSessionTable) \
        .filter(ChatSessionTable.session_id == session_id) \
        .all()
    
    # 返回格式化的消息列表
    result = []
    for record in chat_messages:
        result.append({
            "id": record.id,
            "role": record.role,
            "content": record.content,
            "create_time": record.create_time,
            "feedback": record.feedback
        })
    return result
```

### 2.3 数据流转图

```
┌─────────────┐
│  用户提问    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│ 1. 保存到 sever.db (user消息)   │
│    append_message2db()          │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ 2. AdvancedSQLiteSession 加载历史上下文 │
│    从 conversations.db 读取:            │
│    - system message                     │
│    - 历史 user 消息                     │
│    - 历史 assistant 消息                │
└──────┬──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ 3. 传给大模型 (包含完整上下文)        │
│    Runner.run_streamed(              │
│      agent,                          │
│      input=content,                  │
│      session=session  ← 历史上下文   │
│    )                                 │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ 4. 大模型生成回复（流式输出）         │
│    SSE 实时返回前端                  │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ 5. 保存回复到两个数据库               │
│    - sever.db (业务查询用)            │
│    - conversations.db (上下文用)      │
└──────────────────────────────────────┘
```

### 2.4 核心设计优势

✅ **对话持久化**: 关系型数据库长期存储，支持查询、统计、导出  
✅ **上下文自动管理**: Agent SDK 的 Session 自动维护对话历史  
✅ **支持多轮对话**: 每次调用都携带完整上下文，保证对话连贯性  
✅ **功能完善**: 支持对话列表、删除、反馈等完整功能  
✅ **双层存储分离**: 
   - `sever.db` 用于业务逻辑（用户查询、管理）
   - `conversations.db` 用于大模型上下文（框架内部管理）
