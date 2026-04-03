from PIL import Image
import requests
#from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from modelscope import ChineseCLIPProcessor, ChineseCLIPModel
import torch
from tqdm import tqdm_notebook
import numpy as np

# 官方 openai clip 不支持中文
# https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-base-patch16
model = ChineseCLIPModel.from_pretrained("./model/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("./model/chinese-clip-vit-base-patch16") # 预处理

image_path = "dog.jpg"
image = Image.open(image_path).convert("RGB")
text_labels = [
    "狗",
    "猫",
    "鸟",
    "猪"
]

inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
print(inputs["input_ids"].shape)
with torch.no_grad():
    outputs = model(**inputs)
    # 获取图像与所有文本的相似度分数
    logits_per_image = outputs.logits_per_image
    # 将分数转换为概率
    probs = logits_per_image.softmax(dim=1)
print("分类标签及概率：")
for label, prob in zip(text_labels, probs[0]):
    print(f"  {label}: {prob:.4f}")

# 获取概率最高的类别索引
predicted_class_idx = probs.argmax()
print(f"\n预测结果：这张图片是 '{text_labels[predicted_class_idx]}'")

