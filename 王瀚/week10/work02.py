import os
import base64
import fitz  # PyMuPDF
from openai import OpenAI

# ================= 配置区域 =================
# 1. API 配置
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 2. 任务配置
PDF_PATH = "example.pdf"          # 本地PDF路径
MODEL_NAME = "qwen-vl-max-latest" # 支持推理的模型名
ENABLE_THINKING = True            # 是否开启思考过程
# ===========================================

def get_pdf_first_page_base64(pdf_path):
    """将 PDF 第一页转换为 Base64 编码的图片字符串"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # 第一页
    # 2.0 倍缩放，保证推理时的文字清晰度
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    img_data = pix.tobytes("jpg")
    doc.close()
    
    # 转换为 Base64 格式
    base64_str = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def run_visual_reasoning():
    # 初始化客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # 准备图片
    try:
        image_url_data = get_pdf_first_page_base64(PDF_PATH)
    except Exception as e:
        print(f"读取PDF失败: {e}")
        return

    # 创建流式请求
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_data},
                    },
                    {
                        "type": "text", 
                        "text": "请解析这张PDF页面的内容。请先输出你的思考分析过程，再给出最终的结构化解析结果。"
                    },
                ],
            },
        ],
        stream=True,
        extra_body={
            "enable_thinking": ENABLE_THINKING,
            "thinking_budget": 81920
        }
    )

    reasoning_content = ""
    answer_content = ""
    is_answering = False

    if ENABLE_THINKING:
        print("\n" + "=" * 20 + " 思考过程 (Reasoning) " + "=" * 20 + "\n")

    for chunk in completion:
        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
        
        # 处理推理过程 (Reasoning)
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            content = delta.reasoning_content
            print(content, end='', flush=True)
            reasoning_content += content
        
        # 处理最终回复 (Content)
        elif hasattr(delta, 'content') and delta.content is not None:
            if not is_answering and delta.content != "":
                print("\n\n" + "=" * 20 + " 最终回答 (Answer) " + "=" * 20 + "\n")
                is_answering = True
            
            content = delta.content
            print(content, end='', flush=True)
            answer_content += content

    print("\n\n" + "=" * 48)

if __name__ == "__main__":
    run_visual_reasoning()
