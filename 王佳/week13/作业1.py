"""
1: 阅读 2412-DeepSeek-V3.pdf 论文，总结 V3 相比 V2 改进；
"""
import fitz
import os
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-7458206891744b7aa46d6f7366fecdd5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def extract_text_from_pdf(pdf_path):
    """
    从PDF提取文本内容
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    print(f"正在提取PDF文本（共{len(doc)}页）...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text += f"\n--- 第 {page_num + 1} 页 ---\n{text}"

    doc.close()
    return full_text


def summarize_with_qwen(text, max_pages=5):
    """
    使用Qwen模型总结文档要点
    """
    # 限制文本长度（取前几页）
    pages = text.split("--- 第")
    limited_text = "--- 第".join(pages[:max_pages + 1])

    prompt = f"""
请总结以下DeepSeek-V3技术文档的核心要点，包括：
1. 总结 V3 相比 V2 改进

文档内容：
{limited_text[:8000]}

请用简洁的markdown格式输出总结。
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    print("\n" + "=" * 50)
    print("正在生成文档总结...")
    print("=" * 50 + "\n")

    completion = client.chat.completions.create(
        model="qwen-max",
        messages=messages,
        stream=True,
    )

    full_response = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_response += content

    return full_response


def save_summary(summary, output_path="DeepSeek-V3要点总结.md"):
    """
    保存总结到文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# DeepSeek-V3 文档要点总结\n\n")
        f.write(summary)
    print(f"\n\n总结已保存到: {output_path}")


if __name__ == "__main__":
    pdf_file = "2412-DeepSeek-V3_billingual.pdf"

    try:
        # 步骤1: 提取文本
        text = extract_text_from_pdf(pdf_file)

        # 步骤2: AI总结
        summary = summarize_with_qwen(text, max_pages=106)

        # 步骤3: 保存结果
        # save_summary(summary)

    except FileNotFoundError:
        print(f"错误：找不到PDF文件 {pdf_file}")
    except Exception as e:
        print(f"发生错误：{e}")
        import traceback

        traceback.print_exc()
