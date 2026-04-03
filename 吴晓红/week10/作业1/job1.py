"""
使用中文 CLIP 模型进行 Zero-Shot 图像分类
对本地小狗图片进行多类别分类
"""

import torch
import numpy as np
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def zero_shot_classification(image_path, candidate_labels, model_name="../models/chinese-clip-vit-base-patch16"):
    """
    对单张图片进行 Zero-Shot 分类
    
    Args:
        image_path: 图片路径 (本地图片)
        candidate_labels: 候选标签列表，如 ["狗", "猫", "兔子", "汽车", "房子", "树"]
        model_name: CLIP 模型路径，默认使用本地模型
    
    Returns:
        dict: 包含分类结果和详细得分
    """
    print(f"正在加载模型: {model_name}")
    
    # 1. 加载模型和处理器
    model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    
    # 2. 准备图像
    print(f"加载图片: {image_path}")
    image = Image.open(image_path)
    
    # 3. 准备文本（候选标签）
    print(f"候选标签: {candidate_labels}")
    
    # 4. 提取图像特征
    with torch.no_grad():
        # 图像处理并提取特征
        image_inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features.data.numpy()
        image_features = normalize(image_features, axis=1)  # 归一化
        
        # 文本处理并提取特征
        text_inputs = processor(text=candidate_labels, return_tensors="pt", padding=True)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features.data.numpy()
        text_features = normalize(text_features, axis=1)  # 归一化
    
    # 5. 计算相似度（余弦相似度）
    similarity_scores = np.dot(image_features, text_features.T)[0]
    
    # 6. 结果处理
    results = []
    for i, label in enumerate(candidate_labels):
        results.append({
            "label": label,
            "score": float(similarity_scores[i]),
            "percentage": float(similarity_scores[i] * 100)  # 转换为百分比
        })
    
    # 按得分从高到低排序
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "image_path": image_path,
        "predicted_label": results[0]["label"],
        "predicted_score": results[0]["score"],
        "all_results": results,
        "top_3": results[:3]
    }

def main():
    """主函数"""
    # 配置
    IMAGE_PATH = "./image/dog.jpg"  # 小狗图片路径
    CANDIDATE_LABELS = [
        "狗", "猫", "兔子", "仓鼠", 
        "汽车", "房子", "树", "花",
        "人", "鸟", "鱼", "蝴蝶",
        "椅子", "桌子", "电脑", "手机"
    ]
    
    print("=" * 60)
    print("CLIP Zero-Shot 图像分类 Demo")
    print("=" * 60)
    
    # 执行分类
    results = zero_shot_classification(IMAGE_PATH, CANDIDATE_LABELS)
    
    # 打印结果
    print(f"\n✅ 图片路径: {results['image_path']}")
    print(f"🎯 预测结果: {results['predicted_label']}")
    print(f"📊 预测得分: {results['predicted_score']:.4f}")
    
    print(f"\n🏆 Top 3 结果:")
    for i, item in enumerate(results["top_3"], 1):
        print(f"  {i}. {item['label']}: {item['score']:.4f} ({item['percentage']:.1f}%)")
    
    print(f"\n📈 所有分类结果:")
    for i, item in enumerate(results["all_results"], 1):
        print(f"  {i:2d}. {item['label']:5s} - {item['score']:.4f} ({item['percentage']:.1f}%)")
    
    print(f"\n🎉 分类完成! 图片最可能是: {results['predicted_label']}")

if __name__ == "__main__":
    main()