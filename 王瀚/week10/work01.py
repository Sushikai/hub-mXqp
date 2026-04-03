import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. 加载预训练模型和处理器
# 使用 OpenAI 的经典版本 clip-vit-base-patch32
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. 准备输入数据
image_path = "dog.jpg" 
image = Image.open(image_path)

# 定义分类的标签（候选类别）
labels = ["a photo of a dog", "a photo of a cat", "a photo of a bird", "a photo of a car"]

# 3. 预处理
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# 4. 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 获取结果
# logits_per_image 是图像与每个文本标签的相似度分数
logits_per_image = outputs.logits_per_image 
probs = logits_per_image.softmax(dim=1) # 转化为概率分布

# 6. 打印结果
print("分类结果：")
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")

# 找到概率最高的索引
max_idx = torch.argmax(probs, dim=1).item()
print(f"\n模型认为这张图最可能是: {labels[max_idx]}")
