import os
from openai import OpenAI
from typing import List


api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
client = OpenAI(
    api_key=api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # 使用阿里云百炼的兼容接口
)

# ==========================================
# 1. 定义两个知识库 (模拟向量数据库中的文档)
# ==========================================
ML_KB = [
    "机器学习是人工智能的一个分支，主要研究计算机系统如何利用数据和经验来改善自身的性能。",
    "监督学习是一种机器学习方法，其中模型在标记的数据集上进行训练。",
    "常见的机器学习算法包括线性回归、决策树、支持向量机和随机森林。",
    "无监督学习用于寻找未标记数据中的隐藏模式或内在结构，例如聚类算法K-Means。"
]

LLM_KB = [
    "大语言模型（LLM）是基于深度学习的自然语言处理模型，具有数十亿甚至上百亿个参数。",
    "Transformer架构是大多数现代大语言模型的基础，它引入了自注意力机制。",
    "ChatGPT是基于GPT系列大模型微调而成的对话系统。",
    "大模型可以通过RAG（检索增强生成）技术结合外部知识库，减少幻觉并提供更准确的回答。"
]

# ==========================================
# 2. 检索模块 (RAG中的Retrieval)
# ==========================================
def simple_retrieve(query: str, kb: List[str]) -> str:
    """
    一个简单的检索函数：计算query和KB中每条知识的字面重合度（简单模拟向量检索过程）。
    实际生产环境中应使用向量数据库（如Chroma, FAISS）和Embedding模型进行语义检索。
    """
    best_match = ""
    max_overlap = -1
    
    query_chars = set(query)
    for doc in kb:
        overlap = len(query_chars.intersection(set(doc)))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = doc
            
    return best_match

# ==========================================
# 3. 意图识别模块 (Intent Recognition)
# ==========================================
def detect_intent(query: str) -> int:
    """
    意图识别：
    1: 非机器学习 / 非大模型
    2: 机器学习提问
    3: 大模型提问
    """
    prompt = f"""
请分析以下用户的提问，将其归类为以下三种意图之一，并仅返回意图编号（1、2或3）：
1: 非机器学习、非大模型相关的普通提问（如日常问候、通用常识、其他领域等）
2: 机器学习相关的提问（如传统算法、模型训练、监督学习等）
3: 大语言模型（LLM）相关的提问（如GPT、Transformer、大模型应用等）

用户提问："{query}"
意图编号：
"""
    try:
        # 尝试调用LLM进行意图识别
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        if "2" in result:
            return 2
        elif "3" in result:
            return 3
        else:
            return 1
    except Exception as e:
        # 如果没有配置API Key或网络不通，则降级为简单的关键词匹配，以保证流程可演示
        if "大模型" in query or "LLM" in query or "GPT" in query or "Transformer" in query:
            return 3
        elif "机器" in query or "算法" in query or "监督" in query or "回归" in query:
            return 2
        else:
            return 1

# ==========================================
# 4. 回答生成模块 (RAG中的Generation)
# ==========================================
def generate_answer(query: str, context: str = "") -> str:
    """
    根据给定的上下文和问题生成回答。如果context为空，则直接回答。
    """
    if context:
        prompt = f"请根据以下参考资料回答用户的问题。\n\n参考资料：\n{context}\n\n用户问题：{query}"
    else:
        prompt = f"请回答以下用户问题：\n{query}"
        
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 如果没有API Key，返回一个Mock结果演示流程
        print(f"\n[Error] API调用失败: {e}")
        if context:
            return f"[Mock大模型回答] 根据参考资料({context[:15]}...)，您的答案是：... (请配置API Key以获取真实回答)"
        else:
            return f"[Mock大模型回答] 对于您的问题“{query}”，我的回答是：... (请配置API Key以获取真实回答)"

# ==========================================
# 5. 主流程 (Pipeline)
# ==========================================
def chat_pipeline(query: str):
    print(f"\n{'-'*40}")
    print(f"👤 用户提问: {query}")
    
    # 步骤 1. 意图识别
    intent = detect_intent(query)
    intent_names = {1: "非机器学习/非大模型", 2: "机器学习提问", 3: "大模型提问"}
    print(f"🔍 识别意图: 意图 {intent} ({intent_names[intent]})")
    
    # 步骤 2. 路由与检索生成
    if intent == 1:
        print("⚙️  执行动作: 意图1 -> 直接调用LLM回答 (无检索)")
        answer = generate_answer(query)
        print(f"🤖 AI 回答: {answer}")
        
    elif intent == 2:
        print("⚙️  执行动作: 意图2 -> 检索【机器学习知识库】并结合上下文回答")
        context = simple_retrieve(query, ML_KB)
        print(f"📚 检索结果: {context}")
        answer = generate_answer(query, context)
        print(f"🤖 AI 回答: {answer}")
        
    elif intent == 3:
        print("⚙️  执行动作: 意图3 -> 检索【大模型知识库】并结合上下文回答")
        context = simple_retrieve(query, LLM_KB)
        print(f"📚 检索结果: {context}")
        answer = generate_answer(query, context)
        print(f"🤖 AI 回答: {answer}")

if __name__ == "__main__":
    print("🚀 启动意图识别 + RAG 问答测试流程...")
    
    # 测试样例1：非机器学习/大模型
    chat_pipeline("你好，请问今天适合出门旅游吗？")
    
    # 测试样例2：机器学习
    chat_pipeline("你能解释一下什么是监督学习吗？")
    
    # 测试样例3：大模型
    chat_pipeline("大模型里面的Transformer架构是什么？")
