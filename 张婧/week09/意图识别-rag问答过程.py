import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json

# 假设使用OpenAI API
os.environ["OPENAI_API_KEY"] = "your-api-key"

class IntentRAGSystem:
    def __init__(self):
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 初始化LLM
        self.llm = OpenAI(temperature=0.7, max_tokens=500)
        
        # 初始化知识库
        self.ml_knowledge_base = None
        self.llm_knowledge_base = None
        
        # 构建知识库
        self.build_knowledge_bases()
        
        # 意图识别提示词
        self.intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            请判断以下用户提问的意图类别，只能从以下三类中选择一个：
            1. 非机器学习/非大模型：提问内容与机器学习和大模型都无关
            2. 机器学习提问：提问内容与机器学习相关（包括传统机器学习算法、模型、应用等）
            3. 大模型提问：提问内容与大模型相关（包括大语言模型、LLM、Transformer、GPT等）
            
            用户提问：{query}
            
            只输出意图类别编号（1、2或3），不要输出其他内容：
            """
        )
    
    def build_knowledge_bases(self):
        """构建两个知识库"""
        # 创建示例知识库文档
        ml_documents = [
            "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习。",
            "监督学习是机器学习的一种方法，使用标记数据训练模型，如分类和回归问题。",
            "决策树是一种常用的机器学习算法，通过树状结构进行决策。",
            "随机森林是集成学习算法，通过构建多个决策树来提高预测准确性。",
            "支持向量机(SVM)是一种强大的分类算法，通过寻找最优超平面来分隔数据。",
            "梯度下降是机器学习中常用的优化算法，用于最小化损失函数。",
            "神经网络是受人脑启发的机器学习模型，由多层神经元组成。",
            "聚类是一种无监督学习方法，用于将相似的数据点分组。",
            "交叉验证是评估机器学习模型性能的重要技术。",
            "特征工程是机器学习中从原始数据提取有用特征的过程。"
        ]
        
        llm_documents = [
            "大语言模型(LLM)是基于海量数据训练的大型神经网络模型，如GPT系列。",
            "Transformer是大模型的核心架构，使用自注意力机制处理序列数据。",
            "BERT是一种基于Transformer的预训练语言模型，用于双向上下文理解。",
            "GPT(生成式预训练变换器)是一种自回归语言模型，擅长文本生成任务。",
            "大模型的训练通常需要数千个GPU和大量数据，成本高昂。",
            "提示工程(Prompt Engineering)是优化大模型输入以获得更好输出的技术。",
            "上下文学习(In-Context Learning)是大模型通过学习提示中的示例来执行任务的能力。",
            "大模型微调是在特定任务上对预训练模型进行进一步训练的过程。",
            "RLHF(基于人类反馈的强化学习)是训练大模型与人类偏好对齐的关键技术。",
            "大模型存在幻觉问题，可能生成看似合理但不准确的内容。"
        ]
        
        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len
        )
        
        # 构建机器学习知识库
        ml_splits = text_splitter.split_text("\n".join(ml_documents))
        ml_metadatas = [{"source": "ml_knowledge"} for _ in ml_splits]
        self.ml_knowledge_base = FAISS.from_texts(
            ml_splits, 
            self.embeddings, 
            metadatas=ml_metadatas
        )
        
        # 构建大模型知识库
        llm_splits = text_splitter.split_text("\n".join(llm_documents))
        llm_metadatas = [{"source": "llm_knowledge"} for _ in llm_splits]
        self.llm_knowledge_base = FAISS.from_texts(
            llm_splits, 
            self.embeddings, 
            metadatas=llm_metadatas
        )
        
        print("知识库构建完成！")
    
    def recognize_intent(self, query):
        """识别用户提问的意图"""
        try:
            # 使用LLM进行意图识别
            response = self.llm(self.intent_prompt.format(query=query))
            intent = int(response.strip())
            return intent
        except:
            # 如果LLM识别失败，使用关键词进行简单识别
            return self.keyword_intent_recognition(query)
    
    def keyword_intent_recognition(self, query):
        """基于关键词的意图识别（备用方案）"""
        query_lower = query.lower()
        
        # 机器学习相关关键词
        ml_keywords = ['机器学习', '监督学习', '无监督学习', '决策树', '随机森林', 
                      'svm', '神经网络', '聚类', '回归', '分类', '特征工程',
                      '梯度下降', '交叉验证', '过拟合', '欠拟合']
        
        # 大模型相关关键词
        llm_keywords = ['大模型', '大语言模型', 'llm', 'gpt', 'bert', 'transformer',
                       '提示工程', '微调', '上下文学习', 'rlhf', '自注意力',
                       '预训练', '生成式']
        
        # 判断意图
        is_ml = any(keyword in query_lower for keyword in ml_keywords)
        is_llm = any(keyword in query_lower for keyword in llm_keywords)
        
        if is_ml and not is_llm:
            return 2
        elif is_llm and not is_ml:
            return 3
        elif is_ml and is_llm:
            # 如果同时包含两种关键词，根据提问内容判断
            if '大模型' in query_lower or 'llm' in query_lower or 'gpt' in query_lower:
                return 3
            else:
                return 2
        else:
            return 1
    
    def retrieve_knowledge(self, query, knowledge_base):
        """从指定知识库检索相关知识"""
        # 检索相似文档
        docs = knowledge_base.similarity_search(query, k=3)
        
        # 提取检索结果
        retrieved_texts = [doc.page_content for doc in docs]
        context = "\n".join(retrieved_texts)
        
        return context
    
    def generate_response(self, query, intent):
        """根据意图生成回答"""
        if intent == 1:
            # 非机器学习/非大模型，直接调用LLM回答
            prompt = f"""用户提问：{query}
            
请直接回答用户的问题，保持友好和专业的态度。"""
            response = self.llm(prompt)
            return response
            
        elif intent == 2:
            # 机器学习提问，检索机器学习知识库
            context = self.retrieve_knowledge(query, self.ml_knowledge_base)
            prompt = f"""基于以下机器学习相关的知识库信息回答用户问题：

知识库信息：
{context}

用户问题：{query}

请基于提供的知识库信息回答问题，如果信息不足以回答，请说明。回答要准确、专业。"""
            response = self.llm(prompt)
            return response
            
        elif intent == 3:
            # 大模型提问，检索大模型知识库
            context = self.retrieve_knowledge(query, self.llm_knowledge_base)
            prompt = f"""基于以下大模型相关的知识库信息回答用户问题：

知识库信息：
{context}

用户问题：{query}

请基于提供的知识库信息回答问题，如果信息不足以回答，请说明。回答要准确、专业。"""
            response = self.llm(prompt)
            return response
    
    def process_query(self, query):
        """处理用户查询的完整流程"""
        print(f"\n用户提问：{query}")
        
        # 1. 意图识别
        intent = self.recognize_intent(query)
        intent_names = {1: "非机器学习/非大模型", 2: "机器学习提问", 3: "大模型提问"}
        print(f"识别意图：{intent_names[intent]}")
        
        # 2. 生成回答
        response = self.generate_response(query, intent)
        
        # 3. 输出结果
        print(f"\n回答：{response}")
        print("-" * 80)
        
        return response

def main():
    # 初始化系统
    rag_system = IntentRAGSystem()
    
    # 测试用例
    test_queries = [
        # 意图1：非机器学习/非大模型
        "今天天气怎么样？",
        "什么是人工智能？",  # 这个可能被识别为机器学习，但作为示例
        
        # 意图2：机器学习提问
        "什么是决策树算法？",
        "监督学习和无监督学习有什么区别？",
        "如何防止过拟合？",
        
        # 意图3：大模型提问
        "什么是大语言模型？",
        "Transformer的核心机制是什么？",
        "如何优化大模型的提示工程？"
    ]
    
    # 处理所有测试查询
    for query in test_queries:
        rag_system.process_query(query)
    
    # 交互式问答模式
    print("\n" + "="*80)
    print("进入交互式问答模式（输入 'quit' 退出）")
    print("="*80)
    
    while True:
        user_input = input("\n请输入您的问题：")
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
        
        rag_system.process_query(user_input)

if __name__ == "__main__":
    main()
