import os
import base64
from io import BytesIO
import fitz  # PyMuPDF
import dashscope
from dashscope import MultiModalConversation
from http import HTTPStatus

dashscope.api_key = "sk-b8d6efe8169b4351a91a13b8fe8fd99a"
# 本地 PDF 文件路径
pdf_path = "test_document.pdf"


def pdf_page_to_base64(pdf_file_path, page_num=0):
    """
    使用 PyMuPDF (fitz) 读取 PDF 指定页并转为 Base64
    """
    try:
        # 1. 打开 PDF 文件
        doc = fitz.open(pdf_file_path)
        # 2. 获取指定页面 (page_num 从 0 开始)
        if page_num >= len(doc):
            print(f"❌ 页码超出范围，该 PDF 只有 {len(doc)} 页")
            return None
        page = doc[page_num]
        # 3. 将页面渲染为图片
        # matrix 控制缩放比例，2x2 表示 2倍分辨率，更清晰
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        # 4. 转换为 PNG 字节数据
        img_data = pix.tobytes("png")
        # 5. 编码为 Base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        doc.close()
        return img_base64
    except Exception as e:
        print(f"❌ PDF 转换失败: {e}")
        return None


def analyze_pdf_with_qwen(image_base64):
    """
    调用 Qwen-VL-Max 模型分析图片
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{image_base64}"},
                {"text": "请详细解析这张图片中的内容。如果是文档，请提取标题、主要段落大意和关键信息。"}
            ]
        }
    ]

    print("🚀 正在调用云端 Qwen-VL 模型进行分析...")

    try:
        response = MultiModalConversation.call(model='qwen3-vl-flash', messages=messages)

        if response.status_code == HTTPStatus.OK:
            result_text = response.output.choices[0].message.content[0]['text']
            return result_text
        else:
            print(f"❌ 调用失败: 状态码 {response.status_code}")
            print(f"错误信息: {response.message}")
            return None

    except Exception as e:
        print(f"❌ 发生异常: {e}")
        return None


if __name__ == "__main__":
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：找不到文件 {pdf_path}，请检查路径。")
    else:
        # 注意：PyMuPDF 页码从 0 开始，所以 0 代表第一页
        base64_img = pdf_page_to_base64(pdf_path, page_num=0)
        if base64_img:
            analysis_result = analyze_pdf_with_qwen(base64_img)
            if analysis_result:
                print("\n" + "=" * 30)
                print("📄 云端模型解析结果：")
                print("=" * 30)
                print(analysis_result)
                print("=" * 30)
