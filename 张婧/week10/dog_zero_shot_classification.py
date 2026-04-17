import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 1. 加载模型和处理器（使用本地路径或自动下载）
model_name = "./model/chinese-clip-vit-base-patch16"  # 模型路径
model = ChineseCLIPModel.from_pretrained(model_name)
processor = ChineseCLIPProcessor.from_pretrained(model_name)

# 2. 加载本地小狗图片
image_path = "path/to/your/dog_image.jpg"  # 图片路径
image = Image.open(image_path)

# 3. 定义候选类别（中英文均可，这里用中文示例）
candidate_labels = ["狗", "猫", "鸟", "汽车", "房子", "花朵"]

# 4. 文本预处理：可以加前缀提升效果，如 "一张{}的照片"
texts = [f"一张{label}的照片" for label in candidate_labels]

# 5. 前向推理
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像-文本相似度分数
    probs = logits_per_image.softmax(dim=1)       # 转为概率分布

# 6. 输出结果
pred_idx = probs.argmax().item()
print(f"预测类别: {candidate_labels[pred_idx]}")
print(f"各类别概率: {dict(zip(candidate_labels, probs[0].tolist()))}")
