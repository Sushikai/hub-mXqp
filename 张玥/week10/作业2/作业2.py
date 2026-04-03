import os
import base64
import fitz  # PyMuPDF
from openai import OpenAI


def pdf_first_page_to_png(pdf_path: str, output_png: str, zoom: float = 2.0) -> str:
    """
    将 PDF 的第一页渲染成 PNG 图片
    """
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        raise ValueError("PDF 没有页面，无法解析。")

    page = doc[0]
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(output_png)
    doc.close()
    return output_png


def image_to_base64(image_path: str) -> str:
    """
    将图片转为 base64
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    # 1. 本地 PDF 路径
    pdf_path = "./BAT机器学习面试题库.pdf"

    # 2. 第 1 页转成图片后的保存路径
    image_path = "./pdf_page_1.png"

    # 3. 阿里云百炼 API Key，从环境变量读取
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未检测到环境变量 DASHSCOPE_API_KEY，请先配置 API Key。")

    # 4. PDF 第1页转图片
    pdf_first_page_to_png(pdf_path, image_path, zoom=2.0)
    print(f"PDF 第1页已转换为图片: {image_path}")

    # 5. 图片转 base64
    image_base64 = image_to_base64(image_path)

    # 6. 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 7. 调用 Qwen-VL 进行解析
    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请解析这张由PDF第一页转换而来的图片，"
                            "尽量完整提取其中的标题、正文、列表、表格、公式等内容，"
                            "并使用 Markdown 格式输出。"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    )

    result = response.choices[0].message.content
    print("\n===== 解析结果 =====\n")
    print(result)

    # 8. 保存结果到 markdown 文件
    with open("pdf_page_1_result.md", "w", encoding="utf-8") as f:
        f.write(result)

    print("\n解析结果已保存到 pdf_page_1_result.md")


if __name__ == "__main__":
    main()