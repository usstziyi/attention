import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


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

    attn_layer = ScaledDotProductAttention(d_model)
    out, attn_weights = attn_layer(x)

    print("输入形状: ", x.shape)               # (2, 5, 8)
    print("输出形状: ", out.shape)             # (2, 5, 8)
    print("注意力权重形状:", attn_weights.shape) # (2, 5, 5)

    # 检查注意力权重每一行是否归一化（和为 1）
    print("权重行和: ", attn_weights.sum(dim=-1))

    # ================== 可视化注意力权重 ==================
    # attn_weights: (B, T, T)，第 i 行第 j 列表示位置 i 对位置 j 的关注程度
    attn_np = attn_weights.detach().cpu().numpy()

    fig, axes = plt.subplots(1, batch_size, figsize=(4.5 * batch_size, 4))
    axes = [axes] if batch_size == 1 else list(axes)

    for b, ax in enumerate(axes):
        im = ax.imshow(attn_np[b], cmap="viridis", vmin=0.0, vmax=attn_np[b].max())

        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xlabel("Key position (j)")
        ax.set_ylabel("Query position (i)")
        ax.set_title(f"Attention weights (batch {b})")

        # 在每个格子里标数值，方便核对行和为 1
        for i in range(seq_len):
            for j in range(seq_len):
                ax.text(j, i, f"{attn_np[b, i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if attn_np[b, i, j] < attn_np[b].max() * 0.6 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()