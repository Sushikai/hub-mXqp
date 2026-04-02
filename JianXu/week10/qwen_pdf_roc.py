"""
qwen_pdf_roc.py — 使用 Qwen-VL 解析本地 PDF

支持两种推理模式（mode 参数）：
  - "local" : 加载本地 Qwen-VL 权重，离线推理
  - "api"   : 调用阿里云 DashScope 在线服务，无需本地 GPU

流程（两种模式共用）：
  1. 用 pymupdf（fitz）将 PDF 每页渲染为 PIL.Image
  2. 将页面图片逐页送入 Qwen-VL，附带解析指令
  3. 收集每页输出，拼接为完整文档文本

依赖：
  pip install pymupdf transformers torch accelerate   # local 模式
  pip install openai                                   # api 模式
"""

import base64
import io
from pathlib import Path

import fitz                          # pymupdf
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import openai
import os



# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MODEL_DIR = Path("I:/pretrain_models/Qwen/Qwen2-VL-7B-Instruct")   # local 模式：本地模型路径

# api 模式：DashScope 配置
# API Key 申请：https://dashscope.console.aliyun.com/
DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL    = "qwen-vl-max"   # 可选：qwen-vl-max / qwen-vl-plus

# 每页发给模型的解析指令，可按需修改
DEFAULT_PROMPT = (
    "请完整提取这一页的所有文字内容，保持原有段落结构，"
    "表格用文字行描述，公式保留原始符号。"
)


# ---------------------------------------------------------------------------
# PDF → 图片列表
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str | Path, dpi: int = 150) -> list[Image.Image]:
    """
    将 PDF 每页渲染为 PIL.Image。

    Args:
        pdf_path : PDF 文件路径
        dpi      : 渲染分辨率，越高越清晰但推理越慢；150 是速度与质量的平衡点

    Returns:
        按页码顺序排列的 PIL.Image 列表
    """
    doc = fitz.open(str(pdf_path))
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # 72 是 PDF 默认 DPI
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


# ---------------------------------------------------------------------------
# 工具：PIL.Image → base64 data URL（API 模式使用）
# ---------------------------------------------------------------------------

def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """将 PIL.Image 编码为 base64 data URL，用于 API 请求的图片字段"""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# 本地推理（local 模式）
# ---------------------------------------------------------------------------

class QwenVLParser:
    """
    封装 Qwen3-VL 的加载与单页推理。

    使用方式：
        parser = QwenVLParser(MODEL_DIR)
        text = parser.parse_image(image, prompt="请提取文字")
    """

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"加载模型：{model_dir}")
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto",          # 自动分配到可用 GPU/CPU
        )
        self.model.eval()

    def parse_image(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
        """
        对单张页面图片进行解析，返回模型输出文本。

        Args:
            image  : 页面 PIL.Image
            prompt : 发给模型的指令

        Returns:
            模型解析出的文本字符串
        """
        # Qwen2-VL 的 chat 格式：messages 列表，图片放在 content 里
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        # 用 processor 的 apply_chat_template 生成模型输入
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,     # 每页最多生成 token 数，按需调整
                do_sample=False,         # 解析任务用贪心解码，结果更稳定
            )

        # 只取新生成的部分（去掉输入 prompt 对应的 token）
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 在线推理（api 模式）— DashScope OpenAI 兼容接口
# ---------------------------------------------------------------------------

class QwenVLAPIParser:
    """
    通过阿里云 DashScope 调用在线 Qwen-VL 服务，无需本地 GPU。

    使用方式：
        parser = QwenVLAPIParser(api_key="sk-xxx")
        text = parser.parse_image(image, prompt="请提取文字")

    模型选项（DASHSCOPE_MODEL）：
        qwen-vl-max   — 效果最强
        qwen-vl-plus  — 速度更快、费用更低
    """

    def __init__(
        self,
        api_key: str = DASHSCOPE_API_KEY,
        base_url: str = DASHSCOPE_BASE_URL,
        model: str = DASHSCOPE_MODEL,
    ):
        # openai 库通过修改 base_url 兼容 DashScope 接口
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model

    def parse_image(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
        """
        将单页图片发送至 DashScope API，返回解析文本。

        图片以 base64 data URL 形式内嵌在请求体中，无需上传到公网。

        Args:
            image  : 页面 PIL.Image
            prompt : 发给模型的指令
        """
        image_url = _image_to_base64(image)   # 编码为 data:image/jpeg;base64,...

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text",      "text": prompt},
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 主函数：解析完整 PDF（支持 local / api 两种模式）
# ---------------------------------------------------------------------------

def parse_pdf(
    pdf_path: str | Path,
    mode: str = "local",                  # "local" 或 "api"
    model_dir: str | Path = MODEL_DIR,    # local 模式使用
    api_key: str = DASHSCOPE_API_KEY,     # api 模式使用
    prompt: str = DEFAULT_PROMPT,
    dpi: int = 150,
    page_range: tuple[int, int] | None = None,
) -> str:
    """
    解析本地 PDF，返回全文文本。

    Args:
        pdf_path   : PDF 文件路径
        mode       : "local"（本地模型）或 "api"（DashScope 在线服务）
        model_dir  : local 模式下的本地模型目录
        api_key    : api 模式下的 DashScope API Key
        prompt     : 解析指令
        dpi        : 渲染分辨率
        page_range : 只解析部分页，如 (0, 5) 表示前 5 页；None 表示全文

    Returns:
        各页解析文本拼接后的字符串，页间以分隔线隔开
    """
    pdf_path = Path(pdf_path)
    print(f"读取 PDF：{pdf_path}  ({pdf_path.stat().st_size // 1024} KB)")
    print(f"推理模式：{mode}")

    # 渲染页面
    images = pdf_to_images(pdf_path, dpi=dpi)
    if page_range is not None:
        start, end = page_range
        images = images[start:end]
    print(f"共 {len(images)} 页待解析\n")

    # 根据 mode 选择 parser
    if mode == "local":
        parser = QwenVLParser(model_dir)
    elif mode == "api":
        parser = QwenVLAPIParser(api_key=api_key)
    else:
        raise ValueError(f"mode 须为 'local' 或 'api'，got: {mode!r}")

    # 逐页解析
    results = []
    for idx, img in enumerate(images, start=1):
        print(f"  解析第 {idx}/{len(images)} 页 ...", end=" ", flush=True)
        text = parser.parse_image(img, prompt=prompt)
        print("完成")
        results.append(f"[第 {idx} 页]\n{text}")

    return ("\n\n" + "-" * 40 + "\n\n").join(results)


# ---------------------------------------------------------------------------
# 示例入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PDF_PATH = Path("sample.pdf")   # 替换为实际 PDF 路径

    # --- 本地模式 ---
    # result = parse_pdf(
    #     pdf_path=PDF_PATH,
    #     mode="local",
    #     model_dir=MODEL_DIR,
    #     dpi=150,
    #     # page_range=(0, 3),
    # )

    # --- 在线 API 模式 ---
    result = parse_pdf(
        pdf_path=PDF_PATH,
        mode="api",
        api_key=DASHSCOPE_API_KEY,    # 也可直接传字符串 "sk-xxx"
        dpi=150,
        # page_range=(0, 3),
    )

    print(result)

    # 保存到同名 txt
    out_path = PDF_PATH.with_suffix(".txt")
    out_path.write_text(result, encoding="utf-8")
    print(f"\n已保存至 {out_path}")
