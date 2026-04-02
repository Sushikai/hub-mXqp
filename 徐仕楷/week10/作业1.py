from PIL import Image
import torch
from modelscope import ChineseCLIPProcessor, ChineseCLIPModel

# ====================== 配置 ======================
# 如果模型还没下载，会自动从 ModelScope 下载
model_path = "./model/chinese-clip-vit-base-patch16"  # 本地路径，或直接用 "OFA-Sys/chinese-clip-vit-base-patch16"

model = ChineseCLIPModel.from_pretrained(model_path)
processor = ChineseCLIPProcessor.from_pretrained(model_path)

image_path = "dog.jpg"
image = Image.open(image_path).convert("RGB")

# 可选：调整图像大小到模型常用分辨率（有助于稳定性）
image = image.resize((224, 224))  # Chinese-CLIP ViT-B/16 默认训练分辨率接近此值

# ====================== 改进的文本标签 ======================
base_labels = ["狗", "猫", "鸟", "猪"]

# 多模板 ensemble（强烈推荐，能显著提升精度）
templates = [
    "这是一张{}的照片。",
    "一只{}。",
    "照片中是一只{}。",
    "一张{}的特写照片。",
    "{}的图片。"
]

# 生成所有模板 + 标签的组合
text_labels = []
for label in base_labels:
    for template in templates:
        text_labels.append(template.format(label))

print(f"共生成 {len(text_labels)} 个提示文本用于 ensemble")

# ====================== 处理输入 ======================
inputs = processor(
    text=text_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# ====================== 推理 ======================
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # [1, num_texts]

    # softmax 转概率
    probs = logits_per_image.softmax(dim=1)

# ====================== 聚合 ensemble 结果 ======================
# 对每个原始标签，取所有对应模板的平均概率
num_templates = len(templates)
avg_probs = torch.zeros(len(base_labels), device=probs.device)

for i, label in enumerate(base_labels):
    # 该标签对应的所有模板索引
    start_idx = i * num_templates
    avg_probs[i] = probs[0, start_idx:start_idx + num_templates].mean()

# ====================== 输出结果 ======================
print("\n分类标签及平均概率（ensemble 后）：")
for label, prob in zip(base_labels, avg_probs):
    print(f" {label}: {prob:.4f} ({prob * 100:.2f}%)")

# 预测结果
predicted_idx = avg_probs.argmax().item()
print(f"\n预测结果：这张图片最可能是 **{base_labels[predicted_idx]}** "
      f"（置信度 {avg_probs[predicted_idx]:.4f}）")

# Top-3 显示
print("\nTop-3 预测：")
sorted_indices = avg_probs.argsort(descending=True)
for idx in sorted_indices[:3]:
    print(f" {base_labels[idx]}: {avg_probs[idx]:.4f}")