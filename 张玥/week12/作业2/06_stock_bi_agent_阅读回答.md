# 06-stock-bi-agent 阅读回答

## 1. 什么是前后端分离

前后端分离，简单理解就是：

- 前端负责页面展示、用户交互、输入收集
- 后端负责 HTTP 接口、业务逻辑、数据库操作、Agent/MCP 调用
- 两者通过接口通信，而不是把所有逻辑都写在一个页面脚本里

在 `06-stock-bi-agent` 这个项目中，前后端分离体现得比较明确。

### 前端部分

前端主要是 Streamlit 页面，对应的代码位置包括：

- `06-stock-bi-agent/demo/streamlit_demo.py`
- `06-stock-bi-agent/demo/chat/chat.py`
- `06-stock-bi-agent/demo/chat/chat_list.py`

这些文件主要负责：

- 展示聊天页面
- 接收用户输入
- 显示历史聊天记录
- 调用后端接口获取或提交数据

例如在 `demo/chat/chat.py` 中：

- 页面用 `st.chat_message(...)` 展示消息
- 用 `requests.post("http://127.0.0.1:8000/v1/chat/...", ...)` 调用后端接口
- 用 `requests.post("http://127.0.0.1:8000/v1/chat/get?...")` 拉取某个会话的历史消息

这说明前端并不直接执行大模型对话和数据库逻辑，而是通过 HTTP 接口和后端交互。

### 后端部分

后端主要是 FastAPI 服务，对应代码位置包括：

- `06-stock-bi-agent/main_server.py`
- `06-stock-bi-agent/routers/chat.py`
- `06-stock-bi-agent/services/chat.py`

在 `main_server.py` 中，后端通过 FastAPI 组织路由：

- `app.include_router(user_routers)`
- `app.include_router(chat_routers)`
- `app.include_router(data_routers)`
- `app.include_router(stock_routers)`

说明这个项目后端已经按业务模块拆分成多个 API。

其中聊天模块的接口在 `routers/chat.py`，例如：

- `/v1/chat/`
- `/v1/chat/init`
- `/v1/chat/get`
- `/v1/chat/delete`
- `/v1/chat/list`

这些接口由前端调用，后端再去执行业务逻辑。

### 为什么这就是前后端分离

因为在这个项目里：

- 前端只负责“展示什么、怎么交互”
- 后端只负责“如何处理请求、如何存历史、如何调用 Agent/MCP”
- 前端和后端通过 HTTP API 协作

所以它符合典型的前后端分离架构。

如果用 Java 开发的思路类比：

- Streamlit 前端类似一个单独的前端应用
- FastAPI 后端类似 Spring Boot 的 REST API 服务
- `routers` 类似 Controller
- `services` 类似 Service

---

## 2. 历史对话如何存储

这个项目里的历史对话不是只存在一个地方，而是有多层存储。

## 2.1 前端短期缓存：`st.session_state.messages`

代码位置：

- `06-stock-bi-agent/demo/chat/chat.py`

在这个文件里，前端会把当前页面里的消息保存在：

```python
st.session_state.messages
```

它的作用是：

- 页面刷新前，维持当前聊天窗口的上下文显示
- 让用户在当前前端会话里看到已经发送和收到的消息

例如：

```python
st.session_state.messages.append({"role": "user", "content": prompt.text})
st.session_state.messages.append({"role": "assistant", "content": final_text})
```

这说明前端会把当前聊天消息暂时保存在 Streamlit 的会话状态中。

但是这只是前端层的短期缓存，不是主要的持久化存储。

## 2.2 后端业务持久化：`ChatSessionTable` 和 `ChatMessageTable`

代码位置：

- `06-stock-bi-agent/services/chat.py`

在 `services/chat.py` 中，聊天会话和消息会存到关系型数据库中。

从代码可以看到：

- `ChatSessionTable`：存储会话信息
- `ChatMessageTable`：存储每条聊天消息

例如在 `init_chat_session(...)` 中：

- 先创建一个 `ChatSessionTable` 记录会话
- 再创建一个 `ChatMessageTable` 记录 system message

而在后续聊天流程中，还会通过：

- `append_message2db(session_id, "user", content)`

把用户消息追加进数据库。

这说明项目会把聊天记录存入数据库，便于：

- 后续查看历史对话
- 会话切换
- 会话列表展示
- 删除会话
- 反馈管理

## 2.3 Agent 历史状态存储：`AdvancedSQLiteSession`

代码位置：

- `06-stock-bi-agent/routers/chat.py`
- `06-stock-bi-agent/services/chat.py`
- `06-stock-bi-agent/test/test_agent.py`

项目里还使用了：

```python
AdvancedSQLiteSession(
    session_id=session_id,
    db_path="./assert/conversations.db",
    create_tables=True
)
```

它的作用不是前端展示，而是给 OpenAI Agents SDK 保存对话历史状态。

也就是说，这一层更偏向：

- 让 Agent 记住前面的上下文
- 让下一轮调用还能延续之前的对话状态

所以这个项目里，历史对话至少有三层：

1. 前端页面显示缓存：`st.session_state.messages`
2. 业务数据库持久化：`ChatSessionTable`、`ChatMessageTable`
3. Agent 运行时历史状态：`AdvancedSQLiteSession`

---

## 3. 历史对话如何作为下一次大模型输入

核心答案是：

**通过同一个 `session_id` 复用同一个 `AdvancedSQLiteSession`，再在调用 `Runner.run_streamed(..., session=session)` 时由 SDK 自动带入历史对话。**

## 3.1 后端如何把历史状态交给 Agent

代码位置：

- `06-stock-bi-agent/services/chat.py`

在聊天服务中，会先根据 `session_id` 创建或获取：

```python
session = AdvancedSQLiteSession(
    session_id=session_id,
    db_path="./assert/conversations.db",
    create_tables=True
)
```

然后在调用大模型时传入：

```python
result = Runner.run_streamed(agent, input=content, session=session)
```

这一步很关键。

因为只要：

- `session_id` 不变
- `session` 还是同一个会话对象逻辑

那么 SDK 就会自动把这个会话已有的历史消息带入下一次调用。

也就是说，开发者不需要手动把所有历史消息一条条重新拼接进 prompt。

## 3.2 前端如何恢复历史对话显示

代码位置：

- `06-stock-bi-agent/demo/chat/chat.py`
- `06-stock-bi-agent/demo/chat/chat_list.py`

当前端切换到某个历史会话时，会先调用：

```python
requests.post("http://127.0.0.1:8000/v1/chat/get?session_id=" + st.session_state['session_id'])
```

后端返回这个 `session_id` 对应的历史消息后，前端再把它们重新写回：

```python
st.session_state.messages.append({"role": message["role"], "content": message["content"]})
```

这样用户再次打开这个会话时，页面上就能看到之前的聊天记录。

而 `demo/chat/chat_list.py` 负责展示会话列表，并在点击“进入聊天”时把选中的 `session_id` 放进：

```python
st.session_state.session_id = session_id
```

这样前端和后端就能围绕同一个 `session_id` 继续工作。

## 3.3 总结成一句话

这个项目里“历史对话作为下一次大模型输入”的过程是：

1. 前端保留当前 `session_id`
2. 后端根据 `session_id` 获取或创建 `AdvancedSQLiteSession`
3. 调用 `Runner.run_streamed(..., session=session)`
4. SDK 自动把这个 session 对应的历史消息带入下一轮推理

与此同时：

- 前端通过 `/v1/chat/get` 恢复历史消息显示
- 后端通过数据库记录业务层面的聊天消息

所以它既保证了：

- 用户能看到历史聊天内容
- 大模型也能“记住”之前的对话上下文

---

## 4. 结论

### 问题 1：什么是前后端分离？

在这个项目里，前后端分离就是：

- Streamlit 前端负责页面和交互
- FastAPI 后端负责 HTTP 接口、业务逻辑、数据库和 Agent/MCP 调用
- 两者通过 `/v1/chat/`、`/v1/chat/get`、`/v1/chat/list` 等接口通信

### 问题 2：历史对话如何存储，以及如何作为下一次大模型输入？

历史对话的存储分成三层：

1. 前端短期缓存：`st.session_state.messages`
2. 后端业务持久化：`ChatSessionTable`、`ChatMessageTable`
3. Agent 历史状态：`AdvancedSQLiteSession(db_path="./assert/conversations.db")`

历史对话作为下一次大模型输入的方式是：

- 复用同一个 `session_id`
- 创建同一个 `AdvancedSQLiteSession`
- 在 `Runner.run_streamed(..., session=session)` 中传入该 session
- 由 SDK 自动把历史上下文带入下一轮模型调用

