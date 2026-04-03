import os
import sys
import tempfile
from pathlib import Path

# ==================== 依赖安装（只需运行一次）====================
# pip install dashscope pymupdf

try:
    import fitz  # PyMuPDF（推荐，轻量，无需额外 Poppler）
    from dashscope import MultiModalConversation
except ImportError:
    print("请先安装依赖：")
    print("pip install dashscope pymupdf")
    sys.exit(1)

# ==================== 配置区 ====================
# 获取你的 API Key（阿里云百炼控制台 → Model Studio → API Key）
# https://help.aliyun.com/zh/model-studio/get-api-key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 推荐使用环境变量
if not DASHSCOPE_API_KEY:
    DASHSCOPE_API_KEY = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # ←←← 这里填你的真实 Key

# 推荐模型（性能最强，支持文档解析）
MODEL_NAME = "qwen3-vl-plus"  # 或 "qwen3-vl-max"（更强但稍贵）

# 解析提示词（已优化为文档解析最佳效果）
PROMPT = """请以专业的文档解析能力分析这张图片（PDF第一页）。
1. 完整提取所有文字内容（包括标题、正文、页眉页脚、脚注）。
2. 保留原始排版结构，使用 Markdown 格式输出：
   - 标题使用 # ## ### 
   - 段落正常换行
   - 表格请转换成 Markdown 表格
   - 如果有图片/图表，请描述其内容和位置
3. 最后总结本页主要内容（100字以内）。
请直接输出 Markdown，不要添加任何解释。"""


# ==================== PDF 第一页转图片函数 ====================
def pdf_first_page_to_image(pdf_path: str, dpi: int = 300) -> str:
    """将 PDF 第一页转为高清图片（返回临时文件路径）"""
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    # 使用临时文件，避免污染当前目录
    temp_dir = tempfile.mkdtemp()
    image_path = Path(temp_dir) / "pdf_first_page.jpg"

    doc = fitz.open(str(pdf_path))
    page = doc[0]  # 第一页（索引从0开始）

    # 高清渲染（dpi越高识别越准）
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    pix.save(str(image_path))
    doc.close()

    print(f"✅ PDF 第一页已转换为图片：{image_path}（分辨率 {pix.width}×{pix.height}）")
    return str(image_path)


# ==================== 调用 Qwen-VL 主函数 ====================
def parse_pdf_with_qwen_vl(pdf_path: str):
    """完整流程：PDF → 图片 → Qwen-VL 云端解析"""

    # 1. 转换第一页为图片
    image_path = pdf_first_page_to_image(pdf_path)

    # 2. 调用 Qwen-VL 云端 API（支持本地图片路径）
    print("🚀 正在调用云端 Qwen-VL 进行解析（qwen3-vl-plus）...")

    messages = [{
        "role": "user",
        "content": [
            {"image": image_path},  # DashScope SDK 原生支持本地文件路径
            {"text": PROMPT}
        ]
    }]

    try:
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            messages=messages,
            result_format='message',  # 返回结构化结果
            stream=False  # 可改成 True 实现流式输出
        )

        if response.status_code != 200:
            print("❌ API 调用失败：", response.code, response.message)
            return

        # 提取结果
        result = response.output.choices[0].message.content[0]['text']

        print("\n" + "=" * 60)
        print("🎉 Qwen-VL 解析完成！以下是 PDF 第一页解析结果：")
        print("=" * 60 + "\n")
        print(result)
        print("\n" + "=" * 60)

    except Exception as e:
        print("❌ 调用失败：", str(e))
        print("常见原因：1. API Key 错误  2. 余额不足  3. 网络问题")


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法：")
        print("  python qwen_vl_pdf_parser.py 你的pdf文件.pdf")
        print("示例：python qwen_vl_pdf_parser.py ./test.pdf")
        sys.exit(1)

    pdf_file = sys.argv[1]
    parse_pdf_with_qwen_vl(pdf_file)