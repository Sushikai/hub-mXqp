import os
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings  # 仅用于演示，无需真实API Key
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. 模拟数据与知识库构建
# ==========================================

# 模拟机器学习知识库内容
ml_documents = [
    Document(page_content="支持向量机(SVM)是一种监督学习算法，用于分类和回归分析。"),
    Document(page_content="随机森林是通过集成多个决策树来提高预测准确率的算法。"),
    Document(page_content="梯度下降是一种用于最小化损失函数的优化算法。")
]

# 模拟大模型知识库内容
llm_documents = [
    Document(page_content="Transformer 是一种基于自注意力机制的深度学习模型架构。"),
    Document(page_content="RLHF (人类反馈强化学习) 是用于对齐大模型与人类价值观的训练技术。"),
    Document(page_content="Prompt Engineering (提示词工程) 是通过优化输入文本来提升模型输出质量的方法。")
]

# 初始化向量数据库 (使用 FakeEmbeddings 模拟向量化过程，实际使用请换成 OpenAIEmbeddings 等)
embeddings = FakeEmbeddings(size=10)

# 创建两个独立的知识库实例
vectorstore_ml = Chroma.from_documents(documents=ml_documents, embedding=embeddings, collection_name="ml_kb")
vectorstore_llm = Chroma.from_documents(documents=llm_documents, embedding=embeddings, collection_name="llm_kb")


# ==========================================
# 2. 意图识别模块
# ==========================================

def intent_classifier(query: str) -> str:
    """
    模拟意图识别模型。
    实际场景中，这里应该调用一个微调过的分类模型或者使用 LLM 进行 Few-shot 分类。
    """
    query_lower = query.lower()

    # 简单的关键词规则模拟分类器行为 (仅用于演示)
    ml_keywords = ["svm", "随机森林", "梯度下降", "机器学习", "聚类", "回归"]
    llm_keywords = ["transformer", "rlhf", "大模型", "gpt", "提示词", "attention"]

    if any(k in query_lower for k in ml_keywords):
        return "INTENT_ML"  # 意图2：机器学习提问
    elif any(k in query_lower for k in llm_keywords):
        return "INTENT_LLM"  # 意图3：大模型提问
    else:
        return "INTENT_GENERAL"  # 意图1：通用提问


# ==========================================
# 3. RAG 核心逻辑与路由
# ==========================================

class RAGSystem:
    def __init__(self):
        self.ml_retriever = vectorstore_ml.as_retriever(search_kwargs={"k": 1})
        self.llm_retriever = vectorstore_llm.as_retriever(search_kwargs={"k": 1})

        # 模拟 LLM 对象 (实际使用时请替换为 ChatOpenAI 等真实模型)
        self.mock_llm = self._MockLLM()

    class _MockLLM:
        """模拟 LLM 的调用，避免消耗真实 Token"""

        def invoke(self, prompt):
            if "通用回答" in prompt:
                return f"【通用 LLM 回答】: 我没法检索知识库，但我根据预训练知识告诉你：{prompt.split('用户问题: ')[-1]} 是个很有趣的话题。"
            elif "上下文" in prompt:
                context = prompt.split("上下文: ")[1].split("用户问题")[0]
                question = prompt.split("用户问题: ")[-1]
                return f"【RAG 回答】: 根据检索到的资料 ({context.strip()})，关于你的问题 '{question}'，答案如下..."

    def process_query(self, user_query: str):
        print(f"\n用户提问: {user_query}")

        # 步骤 1: 意图识别
        intent = intent_classifier(user_query)
        print(f"识别意图: {intent}")

        # 步骤 2: 路由分发
        if intent == "INTENT_GENERAL":
            # --- 意图 1: 直接调用 LLM ---
            response = self.mock_llm.invoke(f"通用回答 - 用户问题: {user_query}")

        elif intent == "INTENT_ML":
            # --- 意图 2: 检索机器学习知识库 ---
            print("正在检索 [机器学习知识库]...")
            context_docs = self.ml_retriever.invoke(user_query)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            prompt = f"上下文: {context_text} 用户问题: {user_query}"
            response = self.mock_llm.invoke(f"RAG 回答 - {prompt}")

        elif intent == "INTENT_LLM":
            # --- 意图 3: 检索大模型知识库 ---
            print("正在检索 [大模型知识库]...")
            context_docs = self.llm_retriever.invoke(user_query)
            context_text = "\n".join([doc.page_content for doc in context_docs])

            prompt = f"上下文: {context_text} 用户问题: {user_query}"
            response = self.mock_llm.invoke(f"RAG 回答 - {prompt}")

        else:
            response = "无法识别意图，请重新提问。"

        print(f"最终回答: {response}")
        return response


# ==========================================
# 4. 运行测试
# ==========================================

if __name__ == "__main__":
    rag_system = RAGSystem()

    # 测试用例 1: 意图 1 (通用)
    rag_system.process_query("你好，请帮我写一首关于春天的诗。")

    # 测试用例 2: 意图 2 (机器学习)
    rag_system.process_query("支持向量机 SVM 是用来做什么的？")

    # 测试用例 3: 意图 3 (大模型)
    rag_system.process_query("什么是 RLHF？")