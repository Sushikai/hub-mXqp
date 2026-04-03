"""
作业2: 使用云端的Qwen-VL 对本地的pdf（任意pdf的第一页） 进行解析，写一下这个代码；
https://help.aliyun.com/zh/model-studio/visual-reasoning
"""
from openai import OpenAI
import os
import fitz
import base64
from pathlib import Path

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-7458206891744b7aa46d6f7366fecdd5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def pdf_to_image(pdf_path, page_num=0):
    """
    将PDF的指定页转换为图片
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 渲染为PNG图片（缩放比例2.0提高清晰度）
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)

    # 保存图片到临时文件
    output_path = "temp_page.png"
    pix.save(output_path)
    doc.close();

    return output_path

def analyze_pdf_with_qwen(pdf_path, page_num, query="请分析这张图片的内容"):
    """
    使用Qwen-VL 分析 PDF第一页
    """
    # 转换PDF第一页为图片
    image_path = pdf_to_image(pdf_path, page_num)

    # 读取图片并转换为base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # 创建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                },
                {"type": "text", "text": query}
            ]
        }
    ]

    # 发送请求
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=messages,
        stream=True,
    )

    print("\n" + "=" * 20 + "解析结果" + "=" * 20 + "\n")

    full_response = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_response += content

    # 清理临时文件
    os.remove(image_path)

    return full_response


# 使用示例
if __name__ == "__main__":
    # 替换为你的PDF路径
    pdf_file = "./汽车知识手册.pdf"
    page_num = 0

    print("=" * 50)
    print("Qwen-VL PDF 解析器")
    print("=" * 50)

    try:
        # 自定义查询问题
        query = "请详细描述这张图片的内容，包括文字、图表和任何视觉元素。"
        analyze_pdf_with_qwen(pdf_file, page_num, query)
    except FileNotFoundError:
        print(f"错误：找不到 PDF文件 {{pdf_file}}")
    except Exception as e:
        print(f"发生错误：{e}")