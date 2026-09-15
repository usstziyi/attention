"""用 PyTorch 内置注意力算子实现缩放点积自注意力 (Self-Attention)。

与 sa.py 的关系:
- sa.py 手写完整流程: QK^T -> 缩放 -> mask -> softmax -> @V;
- 本文件把核心计算交给 F.scaled_dot_product_attention (SDPA),
  它会自动分派到 flash-attention / memory-efficient / math 等后端, 更快更省显存。

注意: SDPA 只返回输出, 不返回注意力权重。
需要权重可视化时用 need_attn=True 额外算一份 (仅用于观察, 不参与梯度)。

因果屏蔽用 is_causal=True, 不手写 mask 张量。
"""

import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttentionSDPA(nn.Module):
    """
    缩放点积自注意力 (Self-Attention), 底层调用 SDPA
    输入 x: (batch_size, seq_len, d_model)
    输出:   (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        # d_k, d_v 默认与 d_model 相同（自注意力常用设置）
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model

        # 线性变换权重矩阵 W_q, W_k, W_v
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)

        # 缩放因子 1/sqrt(d_k), 只在手动算注意力权重时用到
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x, is_causal=False, need_attn=False):
        # x: (B, T, d_model)
        Q = self.W_q(x)   # (B, T, d_k)
        K = self.W_k(x)   # (B, T, d_k)
        V = self.W_v(x)   # (B, T, d_v)

        # 1. 内置算子一次完成 QK^T/sqrt(d_k) -> softmax -> @V
        #    is_causal=True 会内置一个下三角因果 mask
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=is_causal,
        )  # (B, T, d_v)

        # 2. 可选: 额外算一份注意力权重, 仅用于观察/画图
        attn = None
        if need_attn:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, T, T)
            if is_causal:
                T_q, T_k = scores.shape[-2:]
                causal = torch.ones(T_q, T_k, dtype=torch.bool).tril(diagonal=T_k - T_q)
                print(causal)
                scores = scores.masked_fill(~causal, float('-inf'))
            attn = F.softmax(scores, dim=-1)  # (B, T, T)

        return out, attn


# ================== 测试示例 ==================
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 8

    x = torch.randn(batch_size, seq_len, d_model)  # 模拟 EEG 电极通道特征

    # 因果掩码（causal mask）：位置 i 只能关注 j <= i，未来位置 j > i 被屏蔽
    # 不再手写 torch.tril，直接交给 SDPA 的 is_causal=True
    attn_layer = ScaledDotProductAttentionSDPA(d_model)
    out, attn_weights = attn_layer(x, is_causal=True, need_attn=True)

    print("输入形状: ", x.shape)               # (2, 5, 8)
    print("输出形状: ", out.shape)             # (2, 5, 8)
    print("注意力权重形状:", attn_weights.shape) # (2, 5, 5)

    # 检查注意力权重每一行是否归一化（和为 1）
    print("权重行和: ", attn_weights.sum(dim=-1))

    # 检查因果掩码是否生效：上三角（未来位置）权重应全为 0
    upper = torch.triu(attn_weights, diagonal=1)
    print("上三角(未来位置)最大权重:", upper.abs().max().item())

    # 一致性检查：SDPA 的输出应等于 手动权重 @ V
    ref = attn_weights @ attn_layer.W_v(x)
    print("与 权重@V 的最大误差:", (out - ref).abs().max().item())

    # ================== 可视化注意力权重 (seaborn) ==================
    # attn_weights: (B, T, T)，第 i 行第 j 列表示位置 i 对位置 j 的关注程度
    attn_np = attn_weights.detach().cpu().numpy()

    # 用上下文管理器临时设置样式：只在该 with 块内生效，不会污染全局 rcParams
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=1.0):
        fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 4.2))
        axes = [axes] if batch_size == 1 else list(axes)

        for b, ax in enumerate(axes):
            sns.heatmap(
                attn_np[b],
                ax=ax,
                cmap="BuGn",
                vmin=0.0,
                vmax=attn_np[b].max(),
                annot=True,                  # 格子里标注数值，方便核对行和为 1
                fmt=".2f",
                annot_kws={"size": 8},
                square=True,
                linewidths=0.5,
                linecolor="white",
                xticklabels=range(seq_len),
                yticklabels=range(seq_len),
                cbar=True,
                cbar_kws={"label": "attention weight"},
            )

            ax.set_xlabel("Key position (j)")
            ax.set_ylabel("Query position (i)")
            ax.set_title(f"Attention weights (batch {b})")

        plt.tight_layout()
        # show/savefig 放在 with 内部，确保渲染时样式仍然生效
        plt.savefig("./SA/attention_weights_sdpa.png", dpi=300)
        plt.show()
