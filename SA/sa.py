import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积自注意力 (Self-Attention)
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

        # 缩放因子 1/sqrt(d_k)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        Q = self.W_q(x)   # (B, T, d_k)
        K = self.W_k(x)   # (B, T, d_k)
        V = self.W_v(x)   # (B, T, d_v)

        # 1. 计算 QK^T 并缩放
        # K.transpose(-2, -1): (B, d_k, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, T, T)

        # 2. 可选：掩码（例如 padding mask 或 causal mask）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 3. softmax 得到注意力权重
        attn = F.softmax(scores, dim=-1)  # (B, T, T)

        # 4. 加权求和
        out = torch.matmul(attn, V)       # (B, T, d_v)
        return out, attn


# ================== 测试示例 ==================
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 8

    x = torch.randn(batch_size, seq_len, d_model)  # 模拟 EEG 电极通道特征

    # 因果掩码（causal mask）：位置 i 只能关注 j <= i，未来位置 j > i 被屏蔽
    # torch.tril 保留下三角（含对角线）为 1，其余为 0，与 forward 里 "0 表示屏蔽" 的约定一致
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # (T, T)，可广播到 (B, T, T)

    attn_layer = ScaledDotProductAttention(d_model)
    out, attn_weights = attn_layer(x, mask=causal_mask)

    print("输入形状: ", x.shape)               # (2, 5, 8)
    print("输出形状: ", out.shape)             # (2, 5, 8)
    print("注意力权重形状:", attn_weights.shape) # (2, 5, 5)

    # 检查注意力权重每一行是否归一化（和为 1）
    print("权重行和: ", attn_weights.sum(dim=-1))

    # 检查因果掩码是否生效：上三角（未来位置）权重应全为 0
    upper = torch.triu(attn_weights, diagonal=1)
    print("上三角(未来位置)最大权重:", upper.abs().max().item())

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
        plt.savefig("./SA/attention_weights.png", dpi=300)
        plt.show()