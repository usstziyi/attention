import torch
import torch.nn as nn
import math


def multi_head_self_attention(x, num_heads=2):
    """
    多头自注意力计算
    参数:
        x: 输入张量, 形状为 (batch_size, seq_len, d_model)
        num_heads: 注意力头数, 默认 2
    返回:
        output: 注意力输出, 形状为 (batch_size, seq_len, d_model)
        attn_weights: 注意力权重, 形状为 (batch_size, num_heads, seq_len, seq_len)
    """
    batch_size, seq_len, d_model = x.shape

    # 每个头的维度: d_k = d_v = d_model / num_heads
    assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
    d_k = d_model // num_heads
    d_v = d_model // num_heads

    # 1. 定义权重矩阵
    # 一次性生成所有头的 Q/K/V 投影权重: (d_model, d_model)
    Wq = torch.randn(d_model, d_model)   # (10, 10)
    Wk = torch.randn(d_model, d_model)   # (10, 10)
    Wv = torch.randn(d_model, d_model)   # (10, 10)
    Wo = torch.randn(d_model, d_model)   # 输出投影 (10, 10)

    # 2. 线性投影得到 Q, K, V
    # x: (B, L, d_model) @ Wq: (d_model, d_model) -> (B, L, d_model)
    Q = x @ Wq    # (5, 8, 10)
    K = x @ Wk    # (5, 8, 10)
    V = x @ Wv    # (5, 8, 10)

    # 3. 拆分多头: (B, L, d_model) -> (B, L, num_heads, d_k) -> (B, num_heads, L, d_k)
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)  # (5, 2, 8, 5) # (8, 5)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)  # (5, 2, 8, 5)
    V = V.view(batch_size, seq_len, num_heads, d_v).transpose(1, 2)  # (5, 2, 8, 5)

    print("Q.shape:", Q.shape)
    print("K.shape:", K.shape)
    print("V.shape:", V.shape)

    # 4. 计算注意力分数 scores = Q @ K^T / sqrt(d_k)
    # Q: (B, H, L, d_k) @ K^T: (B, H, d_k, L) -> (B, H, L, L)
    # (8, 5) @ (5, 8) -> (8, 8)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (5, 2, 8, 8)
    print("scores.shape:", scores.shape)

    # 5. Softmax 归一化得到注意力权重
    attn_weights = torch.softmax(scores, dim=-1)        # (5, 2, 8, 8)
    print("attn_weights.shape:", attn_weights.shape) 

    # 6. 用注意力权重加权求和 V
    # (B, H, L, L) @ (B, H, L, d_v) -> (B, H, L, d_v)
    output = attn_weights @ V                            # (5, 2, 8, 5)
    print("output(heads).shape:", output.shape)

    # 7. 合并多头: (B, H, L, d_v) -> (B, L, H, d_v) -> (B, L, d_model)
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # (5, 8, 10)
    print("output(merged).shape:", output.shape)

    # 8. 输出投影
    # (8, 10) @ (10, 10) -> (8, 10)
    output = output @ Wo                                 # (5, 8, 10)
    print("output(projected).shape:", output.shape)

    return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)   # 便于复现
    x = torch.randn(5, 8, 10)   # (batch=5, seq_len=8, d_model=10)
    output, attn = multi_head_self_attention(x, num_heads=2)
    print("\n最终输出 output:\n", output.shape)
    print("\n注意力权重 attn:\n", attn.shape)