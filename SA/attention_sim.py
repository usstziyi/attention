"""缩放点积注意力 (Scaled Dot-Product Attention) 的模拟实现与形状测试。

公式:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, is_causal=False, dropout_p=0.0):
    """手写实现缩放点积注意力。

    q: (..., L_q, d_k)
    k: (..., L_k, d_k)
    v: (..., L_k, d_v)
    mask: 布尔张量, True 表示保留, 形状需可广播到 (..., L_q, L_k)
    返回: (输出 (..., L_q, d_v), 注意力权重 (..., L_q, L_k))
    """
    d_k = q.size(-1)

    # 1) 相似度打分: (..., L_q, d_k) @ (..., d_k, L_k) -> (..., L_q, L_k)
    scores = q @ k.transpose(-2, -1)

    # 2) 缩放, 防止点积随 d_k 增大而方差爆炸导致 softmax 梯度消失
    scores = scores / math.sqrt(d_k)

    # 3) 可选 mask: 屏蔽 padding 或未来位置
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

    if is_causal:
        L_q, L_k = scores.shape[-2], scores.shape[-1]
        # 下三角(含偏移), 允许第 i 个 query 关注 key 的前 L_k - L_q + i + 1 个位置
        causal = torch.ones(L_q, L_k, dtype=torch.bool).tril(diagonal=L_k - L_q)
        scores = scores.masked_fill(~causal, float("-inf"))

    # 4) 按最后一维做 softmax, 得到每行和为 1 的注意力权重
    attn = torch.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    # 5) 用权重对 V 加权求和: (..., L_q, L_k) @ (..., L_k, d_v)
    out = attn @ v
    return out, attn


def run_case(name, q_shape, k_shape, v_shape, is_causal=False):
    """用随机数据跑一个 case, 校验并打印形状。"""
    q = torch.randn(*q_shape, dtype=torch.float64)
    k = torch.randn(*k_shape, dtype=torch.float64)
    v = torch.randn(*v_shape, dtype=torch.float64)

    out, attn = scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    expected_out = (*q_shape[:-1], v_shape[-1])
    expected_attn = (*q_shape[:-2], q_shape[-2], k_shape[-2])
    assert tuple(out.shape) == expected_out, f"{name}: out {tuple(out.shape)} != {expected_out}"
    assert tuple(attn.shape) == expected_attn, f"{name}: attn {tuple(attn.shape)} != {expected_attn}"

    # 注意力权重每一行应归一化到 1
    row_sum = attn.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-12), f"{name}: softmax 未归一化"

    # 与 PyTorch 内置实现对比(内置实现至少需要 3 维, 2D 时临时补一个 batch 维)
    squeeze = q.dim() == 2
    qr, kr, vr = (t.unsqueeze(0) for t in (q, k, v)) if squeeze else (q, k, v)
    ref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=is_causal)
    if squeeze:
        ref = ref.squeeze(0)
    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-12, f"{name}: 与内置实现不一致, max_diff={max_diff}"

    print(f"[OK] {name}")
    print(f"     q {tuple(q.shape)} @ k^T {tuple(k.transpose(-2, -1).shape)}"
          f" -> scores {tuple(attn.shape)}")
    print(f"     attn {tuple(attn.shape)} @ v {tuple(v.shape)} -> out {tuple(out.shape)}")
    print(f"     行和={row_sum.flatten()[0].item():.12f}  max|out-ref|={max_diff:.3e}")


def main():
    torch.manual_seed(0)

    print("=" * 78)
    print("Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V")
    print("=" * 78)

    # 单头: (L_q, d_k) x (L_k, d_k) x (L_k, d_v), L_q=5, L_k=7
    run_case("单头 2D        L_q=5, L_k=7, d_k=8,  d_v=12",
             (5, 8), (7, 8), (7, 12))

    # 带 batch 维: (B, L, d)
    run_case("批量 3D        B=2, L_q=5, L_k=7, d_k=8, d_v=12",
             (2, 5, 8), (2, 7, 8), (2, 7, 12))

    # 多头: (B, H, L, d)
    run_case("多头 4D        B=2, H=3, L_q=5, L_k=7, d_k=16, d_v=24",
             (2, 3, 5, 16), (2, 3, 7, 16), (2, 3, 7, 24))

    # 自注意力: Q/K/V 来自同一个输入 x, L_q == L_k == L
    x = torch.randn(4, 6, 16, dtype=torch.float64)
    w_q, w_k, w_v = (torch.randn(16, 16, dtype=torch.float64) for _ in range(3))
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    out, attn = scaled_dot_product_attention(q, k, v)
    print(f"[OK] 自注意力       x {tuple(x.shape)} -> q/k/v {tuple(q.shape)}"
          f" -> out {tuple(out.shape)}")
    assert tuple(attn.shape) == (4, 6, 6)
    assert tuple(out.shape) == (4, 6, 16)

    # 因果 mask: 只能看到当前位置及之前
    run_case("因果 mask 4D   B=1, H=1, L=5,  d_k=8,  d_v=8", (1, 1, 5, 8),
             (1, 1, 5, 8), (1, 1, 5, 8), is_causal=True)
    q = torch.randn(1, 1, 5, 8, dtype=torch.float64)
    k = torch.randn(1, 1, 5, 8, dtype=torch.float64)
    v = torch.randn(1, 1, 5, 8, dtype=torch.float64)
    _, causal_attn = scaled_dot_product_attention(q, k, v, is_causal=True)
    upper = torch.triu(causal_attn[0, 0], diagonal=1)
    assert upper.abs().max().item() == 0.0, "因果 mask 未屏蔽未来位置"
    print("     上三角(未来位置)权重全为 0:", (upper.abs().max().item() == 0.0))

    print("=" * 78)
    print("全部 shape 断言通过")


if __name__ == "__main__":
    main()
