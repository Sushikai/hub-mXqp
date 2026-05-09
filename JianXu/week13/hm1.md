# DeepSeek-V3 相比 DeepSeek-V2 的核心改进总结

# 一、整体定位变化

| 模型        | 核心目标                | 关键词             |
| ----------- | ----------------------- | ------------------ |
| DeepSeek-V2 | 极致降低训练与推理成本  | MLA、DeepSeekMoE   |
| DeepSeek-V3 | 工业级超大规模 MoE 系统 | FP8、MTP、稳定训练 |

一句话总结：

- V2：提出“低成本 Transformer 架构”
- V3：解决“超大规模 MoE 工业化训练与推理”

---

# 二、V3 最大升级：FP8 训练

## 1. V2 的问题

V2 虽然已经较低成本：

- 但 BF16 显存占用仍然较高
- 大规模 MoE 通信成本依旧巨大
- GPU 利用率未完全释放

---

## 2. V3 的改进

V3 大规模采用：

- FP8 mixed precision training

即：

- weight
- activation
- gradient

大量使用 FP8 精度。

---

## 3. FP8 的意义

| 精度 | 显存占用 | 训练速度 |
| ---- | -------- | -------- |
| FP32 | 最大     | 最慢     |
| BF16 | 中等     | 较快     |
| FP8  | 最小     | 最快     |

FP8 可以：

- 大幅降低显存
- 提高 TensorCore 吞吐
- 提升 GPU utilization
- 降低训练成本

---

## 4. FP8 最大难点

FP8 极容易：

- loss spike
- overflow
- 梯度爆炸
- 数值不稳定
- nan

V3 最大工程贡献之一：成功稳定训练超大规模 FP8 MoE

---

# 三、MoE（Mixture of Experts）进一步增强

## 1. V2：DeepSeekMoE

V2 已经提出：

# DeepSeekMoE

核心思想：

- shared experts
- routed experts

降低专家冗余。

---

## 2. V3 的增强

V3 重点优化：

### (1) expert load balance

解决：expert collapse

即：

- 所有 token 挤到少数 expert
- GPU 负载不均衡
- 推理效率下降

---

### (2) auxiliary-loss-free balancing

V2 仍较依赖：

- auxiliary balancing loss

V3 则实现：auxiliary-loss-free load balancing

好处：

- 更稳定
- 不干扰主任务 loss
- 专家 specialization 更自然
- MoE 更容易训练

---

# 四、MLA（Multi-head Latent Attention）继续升级

这是 DeepSeek 系列最核心创新之一。

---

# 五、传统 Attention 的问题

Transformer 推理最大瓶颈：KV Cache 太大

因为每层都要保存：

- K
- V

长上下文时显存爆炸。

---

# 六、V2 的 MLA

核心思想：不保存完整 KV

而是：压缩到 latent space

流程：

```text
Token
 ↓
latent compression
 ↓
attention reconstruction
```



# 七、V3 中 MLA 的增强

V3 对 MLA 继续优化：

- 更稳定
- 更高压缩率
- 更好的 attention reconstruction
- 更好的长上下文能力

因此：V3 长文本推理能力明显提升

# 八、引入 Multi-Token Prediction（MTP）

这是 V3 一个重要升级。

## 3. MTP 的意义

Transformer 推理瓶颈：sequential generation

GPU 经常在等待。

MTP 可以：

- 提高 token/s
- 降低 latency
- 提升 speculative decoding 效率