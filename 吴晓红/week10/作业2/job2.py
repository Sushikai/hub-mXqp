"""
使用 Qwen-VL 大模型解析 PDF 文件内容
将 PDF 转换为图像后，使用多模态模型进行分析
只处理第一页内容
"""

import os
import base64
from pathlib import Path
import sys
import subprocess

# 现在导入需要的库
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_image(pdf_path, page_num=0):
    """将 PDF 指定页转换为图像"""
    try:
        # 设置 poppler 路径
        # setup_poppler_path()
        
        print(f"转换 PDF: {pdf_path} 第 {page_num+1} 页")
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        
        if not images:
            raise ValueError("PDF 转换失败，未生成图像")
        
        image = images[0]
        print(f"图像尺寸: {image.size}")
        
        # 保存临时图像文件
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "pdf_page.png"
        image.save(temp_path, "PNG")
        print(f"临时图像保存到: {temp_path}")
        
        return temp_path, image
        
    except Exception as e:
        print(f"PDF 转换错误: {e}")
        return None, None

def image_to_base64(image_path):
    """将图像转换为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_str

def analyze_with_qwen_vl(client, image_path, prompt_text):
    """使用 Qwen-VL 分析图像内容"""
    try:
        # 将图像转换为 base64
        base64_image = image_to_base64(image_path)
        
        # 构建多模态消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]
        
        print("调用 Qwen-VL 模型分析...")
        
        # 调用 API
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 使用 Qwen-VL 模型
            messages=messages,
            stream=False,
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"API 调用错误: {e}")
        return None

def main():
    """主函数"""
    # PDF 文件路径
    pdf_path = Path("image/电脑维修.pdf")
    
    if not pdf_path.exists():
        print(f"错误: PDF 文件不存在: {pdf_path}")
        print(f"当前目录: {Path.cwd()}")
        return
    
    print(f"开始解析 PDF: {pdf_path}")
    print(f"文件大小: {pdf_path.stat().st_size / 1024:.2f} KB")
    
    # 初始化 OpenAI 客户端 (DashScope)
    api_key = "sk-1b0891fe1ab844d98139f95bdd6b402b"  # 使用提供的 API key
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 转换 PDF 第一页为图像
    image_path, pil_image = pdf_to_image(pdf_path, page_num=0)
    print(f"✓ 已提取第一页图像: {image_path}")
    
    if not image_path or not pil_image:
        print("PDF 转换失败，无法继续分析")
        return
    
    # 显示图像预览（可选）
    try:
        pil_image.show()  # 在默认图像查看器中打开
    except:
        pass
    
    # 分析提示词（只分析第一页）
    prompt = """请详细分析这份文档第一页的内容。
    
要求：
1. 提取第一页所有可见的文本内容
2. 描述第一页的布局和结构
3. 识别文档类型（如合同、报告、发票等）
4. 总结第一页的主要内容和目的
5. 使用中文输出分析结果

请确保提取的文本准确完整。"""
    
    # 使用 Qwen-VL 分析
    print("\n" + "="*60)
    print("开始使用 Qwen-VL 模型分析文档内容...")
    print("="*60)
    
    result = analyze_with_qwen_vl(client, image_path, prompt)
    
    if result:
        print("\n" + "="*60)
        print("Qwen-VL 分析结果:")
        print("="*60)
        print(result)
        
        # 将结果保存到文件
        output_file = Path("pdf_analysis_result.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n分析结果已保存到: {output_file}")
    else:
        print("分析失败")
    
    # 清理临时文件
    temp_dir = Path("temp_images")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()
        print(f"已清理临时文件")

if __name__ == "__main__":
    main()