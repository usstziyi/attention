import torch
import math


def single_head_self_attention_with_pad_and_causal_mask(x, pad_mask):
    """
    带 padding mask + causal mask 的单头自注意力

    参数:
        x:        输入张量, 形状 (B, L, d_model)
        pad_mask: padding 掩码, 形状 (B, L)
                  1 / True  表示真实 token
                  0 / False 表示 <pad>
    返回:
        output:       注意力输出, 形状 (B, L, d_v)
        attn_weights: 注意力权重, 形状 (B, L, L)
    """
    B, L, d_model = x.shape
    d_k = 20
    d_v = 20

    Wq = torch.randn(d_model, d_k)
    Wk = torch.randn(d_model, d_k)
    Wv = torch.randn(d_model, d_v)

    Q = x @ Wq   # (B, L, d_k)
    K = x @ Wk   # (B, L, d_k)
    V = x @ Wv   # (B, L, d_v)

    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, L, L)

    pad_mask = pad_mask.bool()                          # (B, L)

    # key_mask: 屏蔽 padding 作为 key 的位置, 形状 (B, 1, L)
    key_mask = ~pad_mask.unsqueeze(1)

    # causal_mask: 下三角为可见, 上三角(不含对角线)为屏蔽, 形状 (L, L)
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=x.device),
        diagonal=1
    )

    # 合并 mask: key 方向屏蔽 OR 因果屏蔽, 广播后形状 (B, L, L)
    combined_mask = key_mask | causal_mask.unsqueeze(0)
    scores_masked = scores.masked_fill(combined_mask, float('-inf'))

    # softmax 得到注意力权重
    attn_weights = torch.softmax(scores_masked, dim=-1)   # (B, L, L)

    # 兜底: 全 pad 句 (整行 -inf) 会产生 NaN
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # query_mask: 清零 padding 作为 query 的行, 形状 (B, L, 1)
    query_mask = pad_mask.unsqueeze(-1)
    attn_weights = attn_weights * query_mask

    # 加权求和
    output = attn_weights @ V                           # (B, L, d_v)

    # 保险: 再次清零 pad query 的输出行
    output = output * query_mask

    return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    B, L, d_model = 2, 5, 10
    x = torch.randn(B, L, d_model)

    # 句子1: 长度 3, 后面 2 个 pad
    # 句子2: 长度 5, 无 pad
    pad_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float32)   # (B, L)

    output, attn = single_head_self_attention_with_pad_and_causal_mask(x, pad_mask)

    print("output.shape:", output.shape)   # (2, 5, 20)
    print("attn.shape:  ", attn.shape)     # (2, 5, 5)

    print("\n句子1 的注意力权重 (上三角为 0, pad 行列也为 0):")
    print(attn[0])

    print("\n句子2 的注意力权重 (上三角为 0):")
    print(attn[1])

    """
    句子1 的注意力权重 (上三角为 0, pad 行列也为 0):
    tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [4.5401e-03, 9.9546e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [3.3166e-11, 2.1967e-12, 1.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

    句子2 的注意力权重 (上三角为 0):
    tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [9.9504e-01, 4.9568e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [9.9987e-01, 1.4935e-12, 1.2676e-04, 0.0000e+00, 0.0000e+00],
            [8.3301e-03, 5.7246e-08, 3.5137e-01, 6.4030e-01, 0.0000e+00],
            [4.2634e-09, 9.9989e-01, 3.1927e-06, 1.0907e-04, 8.5716e-07]])
    """