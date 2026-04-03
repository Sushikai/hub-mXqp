import os
import base64
import fitz  # PyMuPDF
from openai import OpenAI

# 配置您的百炼 API Key
# 建议在运行前配置环境变量，例如： set DASHSCOPE_API_KEY="sk-xxxx"
# 如果没有配置环境变量，也可以直接在这里取消注释并填入您的 Key
# os.environ["DASHSCOPE_API_KEY"] = "您的API_KEY"

def pdf_page_to_base64(pdf_path, page_num=0):
    """
    将本地 PDF 的指定页转换为 base64 编码的图片
    """
    try:
        # 打开 PDF 文件
        doc = fitz.open(pdf_path)
        
        if page_num >= len(doc) or page_num < 0:
            print(f"错误: PDF 只有 {len(doc)} 页，无法提取第 {page_num+1} 页。")
            return None
            
        # 加载指定页
        page = doc.load_page(page_num)
        
        # 将页面渲染为图像，使用较高的分辨率(dpi=200)以保证清晰度
        # zoom=2 大约对应 144 DPI，可以根据需要调整
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # 将图像转换为 PNG 字节流
        img_bytes = pix.tobytes("png")
        
        # 转换为 base64 字符串
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
        
    except Exception as e:
        print(f"处理 PDF 时出错: {e}")
        return None

def analyze_pdf_with_qwenvl(pdf_path, prompt="请详细解析这张图片（PDF第一页）的内容。"):
    """
    使用 Qwen-VL 解析 PDF 页面
    """
    # 检查 API Key
    # 尝试从 DASHSCOPE_API_KEY 或者 ALIYUN_API_KEY 环境变量获取
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
    
    if not api_key:
        print("错误: 未找到 DASHSCOPE_API_KEY 或 ALIYUN_API_KEY 环境变量。")
        print("请先设置环境变量，或者在代码中直接配置 API Key。")
        return
        
    # 1. 提取 PDF 第一页为图片
    print(f"正在提取 '{pdf_path}' 的第一页...")
    img_base64 = pdf_page_to_base64(pdf_path, page_num=0)
    
    if not img_base64:
        return
        
    # 2. 初始化 OpenAI 客户端 (阿里云百炼兼容 OpenAI 接口)
    print("正在调用云端 Qwen-VL 模型进行解析，请稍候...")
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 3. 构建请求并调用模型
    # 使用 qwen-vl-plus 模型，它擅长处理文档和图像
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            stream=True
        )
        
        # 4. 流式输出结果
        print("\n" + "=" * 20 + " 解析结果 " + "=" * 20 + "\n")
        
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end='', flush=True)
                    
        print("\n\n" + "=" * 50)
        
    except Exception as e:
        print(f"\n调用模型时出错: {e}")

if __name__ == "__main__":
    # 获取当前脚本所在目录的父目录，因为 PDF 文件在上一级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # PDF 文件路径
    pdf_file = os.path.join(base_dir, "Week10-多模态大模型.pdf")
    
    if os.path.exists(pdf_file):
        print(f"找到本地 PDF 文件: {pdf_file}")
        
        # 自定义提示词，可以根据需求修改
        custom_prompt = "请提取并解析这一页的全部内容，包括文字、标题以及整体排版结构。"
        
        # 执行解析
        analyze_pdf_with_qwenvl(pdf_file, custom_prompt)
    else:
        print(f"未找到示例文件: {pdf_file}")
        print("请在代码末尾修改 pdf_file 变量，指向您本地实际的 PDF 文件。")
