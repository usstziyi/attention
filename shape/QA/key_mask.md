# 为什么叫 `key_mask`？

## 一、先看它作用在哪个维度

回顾一下注意力分数的形状和含义：

```python
scores = Q @ K.transpose(-2, -1)   # (B, L_q, L_k)
```

| 维度 | 大小 | 含义 |
|------|------|------|
| 第 0 维 | B | batch |
| 第 1 维 | L_q | **query** 位置（"谁在提问"） |
| 第 2 维 | L_k | **key** 位置（"被关注的对象"） |

```
每个句子
              key 方向 (L_k) →
            ┌─────────────────────┐
   query    │  s_00  s_01  s_02 … │
   方向     │  s_10  s_11  s_12 … │
   (L_q)    │  s_20  s_21  s_22 … │
    ↓       └─────────────────────┘
```

`scores[i, j]` 表示：**第 i 个 query 对第 j 个 key 的注意力分数**。

而 pad 是**按 token 位置**存在的——某个位置是 pad，它既可能是 key，也可能是 query。但**在计算注意力分数时，我们要屏蔽的是"作为 key 的 pad 位置"**。

---

## 二、`pad_mask.unsqueeze(1)` 升维后落在哪个维度？

```python
pad_mask:             (B, L)        # L 对应序列位置
pad_mask.unsqueeze(1): (B, 1, L)    # 在中间插入一维
```

对比 `scores (B, L_q, L_k)`：

```
scores:      (B, L_q, L_k)
key_mask:    (B,  1,  L)   ← 广播到 L_k 维度
```

关键点：**`unsqueeze(1)` 插入的维度对应 `L_q`，被广播；保留的最后一维 `L` 对应 `L_k`。**

所以这份 mask **只作用于 key 维度**——它告诉模型："在 key 方向上，哪些位置是 pad，需要屏蔽。"

这就是命名 `key_mask` 的原因：

> **它描述的是"哪些 key 是无效的（pad）"，作用在 scores 的 key 维度（最后一维）上。**

---

## 三、对比：如果换成 `query_mask` 会怎样？

如果我们 `unsqueeze(-1)` 而不是 `unsqueeze(1)`：

```python
query_mask = ~pad_mask.unsqueeze(-1)   # (B, L, 1)
scores = scores.masked_fill(query_mask, float('-inf'))
```

```
scores:       (B, L_q, L_k)
query_mask:   (B, L_q,  1)   ← 广播到 L_q 维度
```

这时 mask 作用在 **query 维度**——含义变成："哪些 query 位置是 pad，把它们的整行分数都屏蔽。" 这对应的是"**无效 query 的行**"。

| 名称 | 写法 | 形状 | 作用维度 | 语义 |
|------|------|------|----------|------|
| `key_mask` | `unsqueeze(1)` | `(B, 1, L)` | 最后一维 (L_k) | 屏蔽无效的 **key** |
| `query_mask` | `unsqueeze(-1)` | `(B, L, 1)` | 中间维 (L_q) | 屏蔽无效的 **query** |

---

## 四、为什么用 `key_mask`（而不是 query_mask）来屏蔽 padding？

标准做法是**只屏蔽 key 方向**，原因：

1. **核心目的**：防止真实 token"注意到"padding。这是 key 方向的问题。
2. **query 方向的处理方式不同**：query 是 pad 时，它的输出我们**事后乘 0 丢弃**（代码里的 `output * pad_mask.unsqueeze(-1)`），而不是在 softmax 前屏蔽整行。
3. **如果屏蔽 query 整行**：整行 scores 全 `-inf` → softmax 分母为 0 → **NaN**。所以 query 方向通常不在 softmax 前处理，而是事后清零。

换句话说：

```
key_mask   → 在 softmax 前屏蔽（防止注意到 pad）  ← 影响 attn_weights
query_mask → 在 softmax 后处理（直接丢弃 pad 输出）← 影响 output
```

---

## 五、一句话总结

> **`key_mask` 这个名字，精确地描述了它的作用：它作用在注意力分数的 key 维度（最后一维）上，用来标记"哪些 key 位置是无效的 pad，不该被 query 关注"。**

命名不是随意的——`unsqueeze(1)` 这个位置决定了它广播到 `L_k` 维度，所以叫 `key_mask` 最贴切。如果写成 `pad_mask.unsqueeze(-1)`，那才应该叫 `query_mask`。