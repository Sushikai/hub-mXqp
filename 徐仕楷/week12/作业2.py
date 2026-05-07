1. 前后端分离架构
后端（FastAPI 服务）

代码位置：main_server.py、routers/、services/、models/

服务地址：http://127.0.0.1:8000

核心任务：处理数据库操作、实现业务逻辑、提供 API 接口

前端（Streamlit 应用）

代码位置：demo/ 目录下的所有 .py 文件

默认地址：http://localhost:8501

核心任务：展示界面、响应用户交互、调用后端 API 获取数据

这种设计让前后端可以独立开发、部署，互不干扰。

2. 数据库与运行时状态管理（双存储设计）
关系型数据库（SQLite）

用途：持久化存储业务数据，如历史消息、用户反馈、会话列表等，支持后续查询和分析。

AdvancedSQLiteSession

用途：管理 Agent 框架的运行时状态，确保每个会话的上下文连续、连贯。

状态隔离

每个 session_id 拥有一个独立的 AdvancedSQLiteSession 实例，不同用户或会话之间的对话状态完全隔离，互不干扰。

自动上下文组装

开发者只需传递 session 对象，无需手动拼接历史消息；Agent 框架会自动读取该 session 的历史记录，构建完整的上下文传递给模型。

流式支持

整个对话流程支持流式输出（Server-Sent Events, SSE），用户能实时看到模型逐字生成的内容，获得类似聊天的流畅体验。