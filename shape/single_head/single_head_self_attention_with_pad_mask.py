import torch
import math


def single_head_self_attention_with_pad_mask(x, pad_mask):
    """
    带 padding mask 的单头自注意力

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

    # 1. 投影矩阵
    Wq = torch.randn(d_model, d_k)
    Wk = torch.randn(d_model, d_k)
    Wv = torch.randn(d_model, d_v)

    Q = x @ Wq   # (B, L, d_k) # pad发生在L维度
    K = x @ Wk   # (B, L, d_k)
    V = x @ Wv   # (B, L, d_v)

    # 2. 注意力分数
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, L, L)

    #先不看B维，先研究一个句子的自注意力分数(L,L)
    """
    S_00,S_01,S_02,S_03,S_04
    S_10,S_11,S_12,S_14,S_15
    S_20,S_21,S_22,S_23,S_24
    S_30,S_31,S_32,S_33,S_34
    S_40,S_41,S_42,S_43,S_44
    """

    # 3. 构造 mask：把 padding 的 key 位置置为 -inf
    #    pad_mask: (B, L) -> (B, 1, L) 才能广播到 (B, L, L)
    #    约定: 1 = 保留, 0 = 屏蔽
    pad_mask = pad_mask.bool()                     # 确保是 bool
    # 让被 mask 的位置为 True, 然后用 masked_fill 填 -inf
    key_mask = ~pad_mask.unsqueeze(1)              # (B, 1, L), True = 要屏蔽key相关位置
    scores_masked = scores.masked_fill(key_mask, float('-inf'))
    # key_mask(L,L)
    """
    False,False,False,-inf,-inf
    False,False,False,-inf,-inf
    False,False,False,-inf,-inf
    False,False,False,-inf,-inf
    False,False,False,-inf,-inf
    """
    # scores_masked(L,L)
    """
    S_00,S_01,S_02,-inf,-inf
    S_10,S_11,S_12,-inf,-inf
    S_20,S_21,S_22,-inf,-inf
    S_30,S_31,S_32,-inf,-inf
    S_40,S_41,S_42,-inf,-inf
    """

    # 4. softmax
    attn_weights = torch.softmax(scores_masked, dim=-1)   # (B, L, L)
    # === 兜底：全 pad 句会产生 NaN ===
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # === query 方向：清零 pad query 的行 ===
    query_mask = pad_mask.unsqueeze(-1)                 # (B, L, 1)
    attn_weights = attn_weights * query_mask            # ← 关键：attn 也洗
    # query_mask(L,L)
    """
    True,True,True,True,True
    True,True,True,True,True
    True,True,True,True,True
    False,False,False,False,False
    False,False,False,False,False
    """
    """
    句子1 (真实长度3, 后2个pad) 的 attn_weights：
            key0   key1   key2   key3   key4
    query0 [ 0.3    0.4    0.3    0.0    0.0 ]   ← 真实 query，正常
    query1 [ 0.3    0.4    0.3    0.0    0.0 ]   ← 真实 query，正常
    query2 [ 0.3    0.4    0.3    0.0    0.0 ]   ← 真实 query，正常
    query3 [ 0.0    0.0    0.0    0.0    0.0 ]   ← pad query
    query4 [ 0.0    0.0    0.0    0.0    0.0 ]   ← pad query
    """

    # === 加权求和 ===
    output = attn_weights @ V                           # (B, L, d_v)
    # 此时 output 的 pad 行天然为 0，无需再乘一次
    # 但为了保险（万一 attn 行没全 0），可以再乘一次
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

    output, attn = single_head_self_attention_with_pad_mask(x, pad_mask)

    print("output.shape:", output.shape)   # (2, 5, 20)
    print("attn.shape:  ", attn.shape)     # (2, 5, 5)

    # 检查: 句子1 中真实 token 对 pad 位置的注意力应为 0
    print("\n句子1 的注意力权重 (最后两列应全为 0,最后两行也全为 0):")
    print(attn[0])

    # 句子2 的注意力权重 (所有位置都正常)
    print("\n句子2 的注意力权重 (所有位置都正常):")
    print(attn[1])

    """
    句子1 的注意力权重 (最后两列应全为 0,最后两行也全为 0):
    tensor([[2.6284e-19, 1.0000e+00, 5.1584e-12, 0.0000e+00, 0.0000e+00],
            [4.5401e-03, 9.9546e-01, 3.1515e-09, 0.0000e+00, 0.0000e+00],
            [3.3166e-11, 2.1967e-12, 1.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

    句子2 的注意力权重 (所有位置都正常):
    tensor([[6.9265e-07, 9.8804e-01, 1.1655e-03, 1.0741e-02, 5.5886e-05],
            [1.4620e-04, 7.2829e-07, 9.9964e-01, 1.6847e-04, 4.8052e-05],
            [9.8036e-01, 1.4644e-12, 1.2428e-04, 1.0554e-09, 1.9516e-02],
            [8.3300e-03, 5.7245e-08, 3.5137e-01, 6.4029e-01, 1.2840e-05],
            [4.2634e-09, 9.9989e-01, 3.1927e-06, 1.0907e-04, 8.5716e-07]])
    """