import torch
import math


def multi_head_self_attention_with_pad_and_causal_mask(x, pad_mask, num_heads=2):
    """
    带 padding mask + causal mask 的多头自注意力

    参数:
        x:         输入张量, 形状 (B, L, d_model)
        pad_mask:  padding 掩码, 形状 (B, L)
                   1 / True  表示真实 token
                   0 / False 表示 <pad>
        num_heads: 注意力头数, 默认 2
    返回:
        output:       注意力输出, 形状 (B, L, d_model)
        attn_weights: 注意力权重, 形状 (B, num_heads, L, L)
    """
    B, L, d_model = x.shape

    assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
    d_k = d_model // num_heads
    d_v = d_model // num_heads  # 这里 d_k == d_v

    # 1. 投影矩阵: 一次性生成所有头的 Q/K/V, 再拆头
    Wq = torch.randn(d_model, d_model)
    Wk = torch.randn(d_model, d_model)
    Wv = torch.randn(d_model, d_model)
    Wo = torch.randn(d_model, d_model)   # 输出投影

    # 2. 线性投影
    Q = x @ Wq   # (B, L, d_model)
    K = x @ Wk   # (B, L, d_model)
    V = x @ Wv   # (B, L, d_model)

    # 3. 拆分多头: (B, L, d_model) -> (B, L, H, d_k) -> (B, H, L, d_k)
    Q = Q.view(B, L, num_heads, d_k).transpose(1, 2)   # (B, H, L, d_k)
    K = K.view(B, L, num_heads, d_k).transpose(1, 2)   # (B, H, L, d_k)
    V = V.view(B, L, num_heads, d_v).transpose(1, 2)   # (B, H, L, d_v)

    # 4. 注意力分数: (B, H, L, d_k) @ (B, H, d_k, L) -> (B, H, L, L)
    #    注意 mask 只作用在最后两个维度 (L, L), 所以可以广播到每个头上
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, H, L, L)

    pad_mask = pad_mask.bool()                          # (B, L)

    # 5. key_mask: 屏蔽 padding 作为 key 的位置
    #    关键: (B, L) -> (B, 1, 1, L), 中间那个 1 是 head 维
    #    如果不显式加 head 维, (B, 1, L) 会和 (B, H, L, L) 把 B 对齐到 H 上 (B == H 时静默出错)
    key_mask = ~pad_mask.view(B, 1, 1, L)               # (B, 1, 1, L), True = 屏蔽

    # 6. causal_mask: 下三角可见, 上三角(不含对角线)屏蔽
    #    (L, L) -> (1, 1, L, L), 只作用在最后两维, 广播到所有 batch / head
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=x.device),
        diagonal=1
    ).view(1, 1, L, L)

    # 7. 合并 mask: key 方向屏蔽 OR 因果屏蔽
    #    (B, 1, 1, L) | (1, 1, L, L) -> (B, 1, L, L), 再广播到 (B, H, L, L)
    combined_mask = key_mask | causal_mask              # (B, 1, L, L)
    scores_masked = scores.masked_fill(combined_mask, float('-inf'))

    # 8. softmax 得到注意力权重
    attn_weights = torch.softmax(scores_masked, dim=-1)   # (B, H, L, L)

    # 兜底: 全 pad 句 (整行 -inf) 会产生 NaN
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # 9. query_mask: 清零 padding 作为 query 的行
    #    关键: (B, L) -> (B, 1, L, 1), head 维同样是 1
    query_mask = pad_mask.view(B, 1, L, 1)              # (B, 1, L, 1)
    attn_weights = attn_weights * query_mask

    # 10. 加权求和: (B, H, L, L) @ (B, H, L, d_v) -> (B, H, L, d_v)
    output = attn_weights @ V

    # 11. 合并多头: (B, H, L, d_v) -> (B, L, H, d_v) -> (B, L, d_model)
    output = output.transpose(1, 2).contiguous().view(B, L, d_model)

    # 12. 输出投影
    output = output @ Wo                                 # (B, L, d_model)

    # 保险: 再次清零 pad query 的输出行
    # 此时 output 已经合并回 (B, L, d_model), 没有 head 维, 正常 (B, L, 1) 广播即可
    output = output * pad_mask.view(B, L, 1)

    return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    B, L, d_model = 2, 5, 10
    num_heads = 2
    x = torch.randn(B, L, d_model)

    # 句子1: 长度 3, 后面 2 个 pad
    # 句子2: 长度 5, 无 pad
    pad_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float32)   # (B, L)

    output, attn = multi_head_self_attention_with_pad_and_causal_mask(x, pad_mask, num_heads)

    print("output.shape:", output.shape)   # (2, 5, 10)
    print("attn.shape:  ", attn.shape)     # (2, 2, 5, 5)

    print("\n句子1 - head0 的注意力权重 (上三角为 0, pad 行列也为 0):")
    print(attn[0, 0])

    print("\n句子1 - head1 的注意力权重 (上三角为 0, pad 行列也为 0):")
    print(attn[0, 1])

    print("\n句子2 - head0 的注意力权重 (上三角为 0):")
    print(attn[1, 0])

    print("\n句子2 - head1 的注意力权重 (上三角为 0):")
    print(attn[1, 1])

    """
    句子1 - head0 的注意力权重 (上三角为 0, pad 行列也为 0):
    tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [9.9996e-01, 4.4388e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [5.2097e-06, 4.5882e-01, 5.4118e-01, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

    句子1 - head1 的注意力权重 (上三角为 0, pad 行列也为 0):
    tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [7.2270e-01, 2.7730e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [1.0000e+00, 1.0737e-14, 6.3063e-16, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

    句子2 - head0 的注意力权重 (上三角为 0):
    tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [1.8070e-05, 9.9998e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [9.9858e-01, 5.7529e-10, 1.4192e-03, 0.0000e+00, 0.0000e+00],
            [9.7372e-01, 4.1159e-07, 6.5124e-03, 1.9764e-02, 0.0000e+00],
            [2.4446e-02, 4.0710e-01, 1.7527e-01, 2.0224e-01, 1.9094e-01]])

    句子2 - head1 的注意力权重 (上三角为 0):
    tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [1.0000e+00, 2.2492e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [2.4239e-04, 7.5204e-01, 2.4771e-01, 0.0000e+00, 0.0000e+00],
            [1.0385e-02, 9.6590e-01, 1.1499e-02, 1.2211e-02, 0.0000e+00],
            [6.7535e-01, 5.0070e-03, 2.5609e-01, 1.1753e-02, 5.1805e-02]])
    """
