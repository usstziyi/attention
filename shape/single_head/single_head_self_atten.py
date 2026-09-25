import torch
import torch.nn as nn
import math


def single_head_self_attention(x):
    """
    单头自注意力计算
    参数:
        x: 输入张量, 形状为 (batch_size, seq_len, d_model)
    返回:
        output: 注意力输出, 形状为 (batch_size, seq_len, d_v)
        attn_weights: 注意力权重, 形状为 (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len, d_model = x.shape
    d_k = 20   # 查询/键的维度
    d_v = 20   # 值的维度

    # 1. 定义权重矩阵 (注意: 输入维度应为 d_model=10)
    Wq = torch.randn(d_model, d_k)   # (10, 20)
    Wk = torch.randn(d_model, d_k)   # (10, 20)
    Wv = torch.randn(d_model, d_v)   # (10, 20)

    # 2. 线性投影得到 Q, K, V
    # x: (B, L, d_model) @ Wq: (d_model, d_k) -> (B, L, d_k)
    Q = x @ Wq    # (5, 8, 20)
    K = x @ Wk    # (5, 8, 20)
    V = x @ Wv    # (5, 8, 20)

    print("Q.shape:", Q.shape)
    print("K.shape:", K.shape)
    print("V.shape:", V.shape)

    # 3. 计算注意力分数 scores = Q @ K^T / sqrt(d_k)
    # K 转置: (B, L, d_k) -> (B, d_k, L)
    # Q @ K^T: (B, L, d_k) @ (B, d_k, L) -> (B, L, L)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (5, 8, 8)
    print("scores.shape:", scores.shape)

    # 4. Softmax 归一化得到注意力权重
    attn_weights = torch.softmax(scores, dim=-1)        # (5, 8, 8)
    print("attn_weights.shape:", attn_weights.shape)

    # 5. 用注意力权重加权求和 V
    # (B, L, L) @ (B, L, d_v) -> (B, L, d_v)
    output = attn_weights @ V                            # (5, 8, 20)
    print("output.shape:", output.shape)

    return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)   # 便于复现
    x = torch.randn(5, 8, 10)   # (batch=5, seq_len=8, d_model=10)
    output, attn = single_head_self_attention(x)
    print("\n最终输出 output:\n", output.shape)
    print("\n注意力权重 attn:\n", attn.shape)
