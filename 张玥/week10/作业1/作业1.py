# clip_zero_shot_demo.py

from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 1. 加载模型和预处理器
model_path = "../model/chinese-clip-vit-base-patch16"
model = ChineseCLIPModel.from_pretrained(model_path)
processor = ChineseCLIPProcessor.from_pretrained(model_path)

# 2. 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# 3. 读取本地图片
image_path = "images/dog.jpg"
image = Image.open(image_path).convert("RGB")

# 4. 构造候选文本类别（中文描述）
texts = [
    "这是一只狗",
    "这是一只猫",
    "这是一只鸟",
    "这是一匹马",
    "这是一辆汽车",
    "这是一只兔子"
]

# 5. 预处理
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

# 6. 放到对应设备
inputs = {k: v.to(device) for k, v in inputs.items()}

# 7. 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图片和文本的相似度
    probs = logits_per_image.softmax(dim=1)      # 转成概率

# 8. 输出结果
print("分类结果：")
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.4f}")

pred_idx = probs[0].argmax().item()
print("\n最终预测：", texts[pred_idx])