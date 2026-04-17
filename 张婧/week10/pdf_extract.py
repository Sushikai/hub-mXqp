import os
import base64
import tempfile
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image


def pdf_first_page_to_image(pdf_path, dpi=200):
    """将PDF的第一页转换为JPEG图片，并返回其路径"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    # 使用 pdf2image 转换PDF第一页
    images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=dpi)
    if not images:
        raise RuntimeError("Failed to convert PDF to image.")
    # 将图片保存到一个临时文件中
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        images[0].save(tmp_file.name, "JPEG")
        return tmp_file.name


def encode_image_to_base64(image_path):
    """将图片文件编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_pdf_with_qwen(pdf_path, prompt="请详细描述这张图片中的内容。", api_key=None, model="qwen-vl-plus"):
    """主函数：处理PDF并调用Qwen-VL模型进行分析"""
    # 1. 准备工作
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Please set the DASHSCOPE_API_KEY environment variable.")

    # 2. 转换PDF为图片
    print(f"正在处理PDF文件: {pdf_path}")
    image_path = pdf_first_page_to_image(pdf_path)
    print(f"PDF第一页已转换为临时图片: {image_path}")

    try:
        # 3. 编码图片
        base64_image = encode_image_to_base64(image_path)
        # 4. 初始化OpenAI风格的客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 5. 调用模型进行分析
        print(f"正在调用模型 {model} 进行分析...")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        # 6. 获取结果
        return completion.choices[0].message.content
    finally:
        # 7. 清理临时文件
        if os.path.exists(image_path):
            os.unlink(image_path)
            print(f"临时文件已清理: {image_path}")


if __name__ == "__main__":
    # --- 在这里设置你的PDF文件路径 ---
    pdf_file_path = "your_document.pdf"
    # --- 可选：自定义你的分析提示词 ---
    custom_prompt = "请提取这张图片中的所有文字信息，并以Markdown格式的表格呈现。"
    # --- 执行分析 ---
    try:
        analysis_result = analyze_pdf_with_qwen(pdf_file_path, prompt=custom_prompt)
        print("\n========== 分析结果 ==========")
        print(analysis_result)
        print("=================================")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
