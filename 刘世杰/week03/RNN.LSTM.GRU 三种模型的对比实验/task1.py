import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ===================== 数据加载与预处理 =====================
dataset = pd.read_csv("dataset.csv", sep=",", header=None, nrows=100, encoding='gbk')
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40
index_to_label = {i: label for label, i in label_to_index.items()}


# ===================== 数据集类 =====================
class CharTextDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 字符转索引 + 填充/截断
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# ===================== 模型定义 =====================
class RNNClassifier(nn.Module):
    """基础RNN分类器"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


class LSTMClassifier(nn.Module):
    """LSTM分类器"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


class GRUClassifier(nn.Module):
    """GRU分类器"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# ===================== 训练与评估函数 =====================
def train_model(model, dataloader, criterion, optimizer, num_epochs=4):
    """通用训练函数"""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if idx % 50 == 0:
                print(f"Batch {idx}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
    return model


def evaluate_accuracy(model, dataloader):
    """评估模型精度"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def classify_text(text, model, char_to_index, max_len, index_to_label):
    """单文本分类预测"""
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_index = torch.max(output, 1)
    return index_to_label[predicted_index.item()]


# ===================== 实验主流程 =====================
if __name__ == "__main__":
    # 1. 初始化数据集和数据加载器
    dataset = CharTextDataset(texts, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. 模型超参数
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)
    lr = 0.001
    num_epochs = 4

    # 3. 定义模型列表
    models = {
        "RNN": RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        "LSTM": LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        "GRU": GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    }

    # 4. 存储实验结果
    results = {}

    # 5. 逐个训练并评估模型
    for model_name, model in models.items():
        print("\n" + "=" * 50)
        print(f"训练 {model_name} 模型")
        print("=" * 50)

        # 初始化损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs)

        # 评估精度
        accuracy = evaluate_accuracy(trained_model, dataloader)
        results[model_name] = accuracy
        print(f"\n{model_name} 模型精度: {accuracy:.2f}%")

        # 测试示例文本
        test_texts = ["帮我导航到北京", "查询明天北京的天气"]
        for text in test_texts:
            pred_label = classify_text(text, trained_model, char_to_index, max_len, index_to_label)
            print(f"输入 '{text}' → 预测标签: '{pred_label}'")

    # 6. 打印对比结果
    print("\n" + "=" * 50)
    print("模型精度对比结果")
    print("=" * 50)
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.2f}%")
