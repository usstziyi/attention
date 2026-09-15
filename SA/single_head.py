import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# =========================
# 1. 定义单头自注意力模块
# =========================
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.0, batch_first=True):
        super().__init__()
        # num_heads=1 即单头
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=batch_first,   # 输入形状为 (batch, seq_len, embed_dim)
        )

    def forward(self, x, need_weights=False, is_causal=False):
        # 因果掩码：位置 i 只能关注 j <= i，未来位置 j > i 被屏蔽
        attn_mask = None
        if is_causal:
            T = x.size(1)
            # nn.MultiheadAttention 的约定：bool 张量里 True 表示"不允许关注"，所以上三角置 True
            attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            print(attn_mask)
            
            """
            tensor([[False,  True,  True,  True,  True],
                    [False, False,  True,  True,  True],
                    [False, False, False,  True,  True],
                    [False, False, False, False,  True],
                    [False, False, False, False, False]])
            """

        # 自注意力：Q = K = V = x
        out, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            is_causal=is_causal,
            need_weights=need_weights,
        )
        return out, attn_weights


# =========================
# 2. 测试
# =========================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    x = torch.randn(batch_size, seq_len, embed_dim)

    model = SingleHeadSelfAttention(embed_dim=embed_dim, batch_first=True)

    # 前向传播，顺便拿到注意力权重（is_causal=True 时只用下三角，看不到未来位置）
    out, attn_weights = model(x, need_weights=True, is_causal=True)

    print("输入 x 形状:      ", x.shape)              # (2, 5, 8)
    print("输出 out 形状:    ", out.shape)            # (2, 5, 8)
    print("注意力权重形状:   ", attn_weights.shape)    # (2, 5, 5)

    # 验证注意力权重每行和为 1（softmax 归一化）
    print("注意力权重行和:   ", attn_weights.sum(dim=-1))

    # 检查因果掩码是否生效：上三角（未来位置）权重应全为 0
    upper = torch.triu(attn_weights, diagonal=1)
    print("上三角(未来位置)最大权重:", upper.abs().max().item())

    # ================== 可视化注意力权重 (seaborn) ==================
    # 单头，所以 attn_weights 就是 (B, T, T)，第 i 行第 j 列表示位置 i 对位置 j 的关注程度
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
        plt.savefig("./SA/attention_weights_single_head.png", dpi=300)
        plt.show()
