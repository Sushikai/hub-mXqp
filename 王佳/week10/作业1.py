"""
1: 本地使用一张图（小狗图片），尝试进行一下clip的zero shot classification 图像分类；
"""
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 加载本地Chinese-CLIP模型
model_path = "e:/models/chinese-clip-vit-base-patch16"


def clip_zero_shot_chinese(image_path):
    """
    使用 Chinese-CLIP进行零样本分类
    """
    # 加载模型和处理器
    model = ChineseCLIPModel.from_pretrained(model_path)
    processor = ChineseCLIPProcessor.from_pretrained(model_path, use_fast=True)

    # 加载图像
    image = Image.open(image_path)

    # 定义中文候选类别
    candidates = [
        "小猫",
        "鸟",
        "汽车",
        "船",
        "人",
        "建筑",
        "树",
        "小狗",
    ]

    # 处理输入
    inputs = processor(
        text=candidates,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image

        # 计算概率
        probs = logits_per_image.softmax(dim=1)

    # 获取结果
    top_idx = probs.argmax().item()

    print(f"预测结果：{candidates[top_idx]}")
    print(f"置信度：{probs[0][top_idx].item():.4f}")

    # 显示所有类别的概率
    for i, (candidate, prob) in enumerate(zip(candidates, probs[0])):
        print(f"{i + 1}. {candidate}: {prob.item():.4f}")


# 使用示例
if __name__ == "__main__":
    image_path = "./小狗图片.jpg"

    print("=" * 50)
    print("Chinese-CLIP Zero-Shot Image Classification")
    print("=" * 50)

    try:
        clip_zero_shot_chinese(image_path)
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")