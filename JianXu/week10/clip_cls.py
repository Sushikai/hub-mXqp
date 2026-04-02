from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

MODEL_DIR = Path("I:/pretrain_models")
DATA_DIR = Path(__file__).parent / "data"

# 1. 加载模型和处理器
model_name = Path(MODEL_DIR / "Clip" / "clip-vit-base-patch32")
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. 读取 data/ 下所有图片
image_paths = sorted(DATA_DIR.glob("*"))
image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
images = [Image.open(p).convert("RGB") for p in image_paths]

# 3. 准备候选类别文本
labels = ["dog", "cat", "bird", "car", "person"]
texts = [f"a photo of a {label}" for label in labels]

# 4. 批量送入 processor，一次矩阵运算
inputs = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: [num_images, num_labels]
    probs = logits_per_image.softmax(dim=1)

# 5. 输出每张图片的预测结果
for img_path, img_probs in zip(image_paths, probs):
    pred_idx = img_probs.argmax().item()
    print(f"\n图片: {img_path.name}")
    print(f"  预测类别: {labels[pred_idx]}  ({img_probs[pred_idx].item():.4f})")
    print("  所有类别概率：")
    for label, prob in zip(labels, img_probs):
        print(f"    {label}: {prob.item():.4f}")