import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


csv_path = "./dataset.csv"
df = pd.read_csv(csv_path, encoding="gbk")

texts = df["text"].tolist()
labels = df["label"].tolist()
# ===================== 标签编码 =====================
lbl = LabelEncoder()
labels = lbl.fit_transform(labels)


x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels,random_state=42,
)

# ===================== 加载BERT模型 =====================
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(lbl.classes_))

# ===================== 编码 =====================
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": y_train
})
test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": y_test
})

# ===================== 评估 =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# ===================== 训练参数 =====================
training_args = TrainingArguments(
    output_dir="./bert_sentiment_model",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# ===================== 训练 =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
print("\n======== 最终测试集评估 ========")
print(trainer.evaluate())

# ===================== 预测函数 =====================
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return lbl.inverse_transform([pred_id])[0]

# ===================== 测试 =====================
print("\n======== 模型预测测试 ========")
test1 = "消食顺气片"
print(f"输入：{test1} → 预测：{predict_text(test1)}")

test2 = "孕妇禁止服用"
print(f"输入：{test2} → 预测：{predict_text(test2)}")

test3 = "紫草（根）、全蝎、连翘、通草、地肤子、滑石、知母、甘草 "
print(f"输入：{test3} → 预测：{predict_text(test3)}")
