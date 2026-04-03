"""
作业：CLIP Zero-Shot 图像分类
使用本地小狗图片进行分类
"""

import os

# 自动设置 Hugging Face 镜像源（解决国内网络下载超时问题）
# 必须在导入 transformers 之前设置环境变量！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用huggingface_hub的网络代理重试，避免没必要的超时等待
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# 解决 OpenMP 多次加载报错问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 获取当前脚本所在目录的绝对路径，确保无论在哪里运行脚本，相对路径都正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 1. 加载模型（带缓存机制） ====================
print("正在加载 Chinese-CLIP 模型...")

# 模型路径（根据week10的代码）
# 如果本地不存在该目录，则自动从 Hugging Face 镜像下载并保存到本地
local_model_path = os.path.join(BASE_DIR, "model", "chinese-clip-vit-base-patch16")

# 检查本地是否已经有下载好的模型文件
has_local_model = False
if os.path.exists(local_model_path):
    # 简单校验文件夹内是否有必要的模型文件（如 config.json）
    if os.path.exists(os.path.join(local_model_path, "config.json")):
        has_local_model = True

if has_local_model:
    model_path = local_model_path
    print(f"✅ 检测到本地已存在模型，直接从本地加载: {model_path}")
else:
    model_path = "OFA-Sys/chinese-clip-vit-base-patch16"
    print(f"⚠️ 本地模型未找到，将从 Hugging Face 自动下载: {model_path}")

# 加载模型和处理器
model = ChineseCLIPModel.from_pretrained(model_path)
processor = ChineseCLIPProcessor.from_pretrained(model_path)

# 如果是首次从线上下载，将其保存到本地，方便后续复用
if not has_local_model:
    print(f"💾 正在将模型保存到本地以供后续复用: {local_model_path}")
    os.makedirs(local_model_path, exist_ok=True)
    model.save_pretrained(local_model_path)
    processor.save_pretrained(local_model_path)
    print("✅ 模型保存完成！")

# 将模型设为评估模式
model.eval()

print("模型加载完成！")

# ==================== 2. 准备候选标签 ====================
# 定义候选类别（针对小狗图片，可以准备多个动物类别）
candidate_labels = [
    "小狗",      # 正确类别
    "小猫",
    "兔子",
    "老虎",
    "狮子",
    "大象",
    "熊猫",
    "小鸟",
    "汽车",
    "飞机"
]

print(f"\n候选类别: {candidate_labels}")

# ==================== 3. 加载本地图片 ====================
def load_image(image_path):
    """加载本地图片"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"加载图片失败: {e}")
        return None

# 自动查找 data 目录下的图片
def find_image(folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    if not os.path.exists(folder):
        return None
    for ext in image_extensions:
        for filename in os.listdir(folder):
            if filename.lower().endswith(ext):
                return os.path.join(folder, filename)
    return None

# 图片路径（自动查找或手动指定）
data_dir = os.path.join(BASE_DIR, "data")
image_path = find_image(data_dir)
if image_path is None:
    image_path = os.path.join(data_dir, "dog.jpg")  # 默认路径

image = load_image(image_path)
if image is None:
    print(f"\n请准备一张小狗图片，放在 {data_dir} 目录下")
    print("支持的格式: JPG, PNG, JPEG, BMP, WEBP")
    exit(1)
else:
    print(f"\n成功加载图片: {image_path}")
    print(f"图片尺寸: {image.size}")

# ==================== 4. Zero-Shot 分类核心代码 ====================
def zero_shot_classify(image, candidate_labels, top_k=3):
    """
    CLIP Zero-Shot 分类

    Args:
        image: PIL Image对象
        candidate_labels: 候选类别列表
        top_k: 返回前k个最可能的类别

    Returns:
        排序后的类别和相似度分数
    """

    # ----- 4.1 处理图像 -----
    # 使用processor处理图像
    image_inputs = processor(images=image, return_tensors="pt")

    # 获取图像特征
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        # 归一化特征向量
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # ----- 4.2 处理文本标签 -----
    # 使用processor处理文本（中文CLIP支持中文标签）
    text_inputs = processor(
        text=candidate_labels,
        return_tensors="pt",
        padding=True
    )

    # 获取文本特征
    with torch.no_grad():
        # 由于transformers中部分版本的ChineseCLIPModel在直接调用get_text_features时存在bug
        # 手动获取文本特征
        text_outputs = model.text_model(**text_inputs)
        # 提取 CLS token 的特征作为池化输出
        pooled_output = text_outputs.last_hidden_state[:, 0, :]
        # 投影到 CLIP 特征空间
        text_features = model.text_projection(pooled_output)
        
        # 归一化特征向量
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ----- 4.3 计算相似度 -----
    # 使用余弦相似度（点积，因为已经归一化）
    similarity_scores = (image_features @ text_features.T).squeeze(0)

    # 转换为概率分布（使用softmax）
    probs = torch.nn.functional.softmax(similarity_scores, dim=0)

    # ----- 4.4 排序并返回结果 -----
    # 按相似度降序排序
    sorted_indices = similarity_scores.argsort(descending=True)

    results = []
    for idx in sorted_indices[:top_k]:
        label = candidate_labels[idx]
        score = similarity_scores[idx].item()
        prob = probs[idx].item()
        results.append({
            "label": label,
            "similarity": round(score, 4),
            "probability": round(prob * 100, 2)
        })

    return results

# ==================== 5. 执行分类并展示结果 ====================
print("\n" + "="*50)
print("开始 Zero-Shot 分类...")
print("="*50)

# 执行分类
results = zero_shot_classify(image, candidate_labels, top_k=5)

# 展示结果
print(f"\n图片: {image_path}")
print("-"*50)
print("预测结果（Top 5）：")
print("-"*50)

for i, result in enumerate(results, 1):
    print(f"{i}. 类别: {result['label']:<6} | "
          f"相似度: {result['similarity']:.4f} | "
          f"概率: {result['probability']:.2f}%")

# 预测类别
predicted_label = results[0]['label']
confidence = results[0]['probability']

print("-"*50)
print(f"[OK] 最终预测: {predicted_label} (置信度: {confidence:.2f}%)")

# ==================== 6. 可视化展示 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：原始图片
axes[0].imshow(image)
axes[0].set_title(f"输入图片\n预测: {predicted_label}", fontsize=14, fontweight='bold')
axes[0].axis('off')

# 右图：分类结果条形图
labels = [r['label'] for r in results]
scores = [r['probability'] for r in results]
colors = ['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(len(labels))]

bars = axes[1].barh(labels, scores, color=colors)
axes[1].set_xlabel('概率 (%)', fontsize=12)
axes[1].set_title('Zero-Shot 分类结果', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 100)

# 在条形上添加数值
for bar, score in zip(bars, scores):
    axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{score:.1f}%', va='center', fontsize=11)

# 反转y轴使最高分数在上面
axes[1].invert_yaxis()

plt.tight_layout()
save_path = os.path.join(BASE_DIR, 'zero_shot_result.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\n结果图已保存为 {save_path}")
