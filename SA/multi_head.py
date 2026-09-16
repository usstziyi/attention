import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# =========================
# 1. 定义多头自注意力模块
# =========================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.0, batch_first=True):
        super().__init__()
        # embed_dim 必须能被 num_heads 整除，每个头的维度为 embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,   # 输入形状为 (batch, seq_len, embed_dim)
        )

    def forward(self, x, need_weights=False, average_attn_weights=False):
        # 因果掩码：位置 i 只能关注 j <= i，未来位置 j > i 被屏蔽
        attn_mask = None
        # 只有在需要注意力权重时才手动创建掩码
        # 这个掩码是给每个头的注意力机制使用的，所以需要广播到每个头的 logits 矩阵
        # 分头的本质是切分QKV的特征维度，不改变时间序列的维度T,而B->B*num_heads这个由显卡自动处理
        # 经过sdpa之后，每个头的注意力权重(B*num_heads,T,T),输出(B*num_heads,T,head_dim)
        # 最后再拼接起来，得到(B,T,embed_dim)
        if need_weights:
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

        """
        分数矩阵 shape: (B * num_heads, T, T)       # 这就是 logits，按头平铺的
        attn_mask (T, T) 广播加到这里                # ← 掩码生效的地方
        softmax(dim=-1)                            # 每个头独立归一化(B*num_heads,T,T)
        attn_output = softmax @ V                  # (B*num_heads, T, head_dim)
        reshape/拼接 → (B, T, embed_dim)            # 此时掩码已用完
        """
        # 自注意力：Q = K = V = x
        # average_attn_weights=False 时保留每个头各自的权重，形状 (B, num_heads, T, T)
        out, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        """
        average_attn_weights只在`need_weights=True` 时才有意义 。
        因为权重是"顺手额外返回"的诊断信息，如果`need_weights=False` ，
        内部走的是`scaled_dot_product_attention` 快速路径，压根不产出权重矩阵，这个参数就被忽略。
        average_attn_weights不参与前向计算 。
        归一化、加权求和、拼接都是逐头独立做好的，`out` 无论这个参数取什么值都完全一样。
        取平均只发生在"要不要把每头的权重合并成一张图"这一步，纯属给外部看/画图用。
        """

        return out, attn_weights


# =========================
# 2. 测试
# =========================
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embed_dim = 8
    num_heads = 4
    head_dim = embed_dim // num_heads

    x = torch.randn(batch_size, seq_len, embed_dim)

    model = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    # 前向传播，顺便拿到每个头各自的注意力权重
    out, attn_weights = model(x, need_weights=True, average_attn_weights=False)

    print("输入 x 形状:        ", x.shape)             # (2, 5, 8)
    print("输出 out 形状:      ", out.shape)           # (2, 5, 8)
    print("注意力权重形状:     ", attn_weights.shape)   # (2, 4, 5, 5)
    print("每个头的维度:       ", head_dim)             # 2

    # 验证注意力权重每行和为 1（softmax 归一化）
    print("注意力权重行和:     ", attn_weights.sum(dim=-1))

    # 检查因果掩码是否生效：上三角（未来位置）权重应全为 0
    # triu只对最后两个维度做上三角，前面的维度全部当作 batch 逐一处理 。
    upper = torch.triu(attn_weights, diagonal=1)
    print("上三角(未来位置)最大权重:", upper.abs().max().item())

    # ================== 可视化注意力权重 (seaborn) ==================
    # 多头，attn_weights 形状为 (B, num_heads, T, T)，第 h 个头里第 i 行第 j 列表示位置 i 对位置 j 的关注程度
    attn_np = attn_weights.detach().cpu().numpy() # shape (2, 4, 5, 5)

    # 用上下文管理器临时设置样式：只在该 with 块内生效，不会污染全局 rcParams
    with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=1.0):
        fig, axes = plt.subplots(
            batch_size,
            num_heads,
            figsize=(3.6 * num_heads, 4.0 * batch_size),
            squeeze=False,
        )

        for b in range(batch_size):
            for h in range(num_heads):
                ax = axes[b][h]
                sns.heatmap(
                    attn_np[b][h],
                    ax=ax,
                    cmap="BuGn",                 # 颜色映射
                    vmin=0.0,                    # 颜色下限
                    vmax=attn_np[b][h].max(),    # 颜色上限，按每个头自身最大值归一
                    annot=True,                  # 格子里标注数值，方便核对行和为 1
                    fmt=".2f",                   # 标注数值保留两位小数
                    annot_kws={"size": 8},       # 标注字体大小
                    square=True,                 # 单元格保持正方形
                    linewidths=0.5,              # 单元格之间的分隔线宽度
                    linecolor="white",           # 分隔线颜色
                    xticklabels=range(seq_len),  # x 轴刻度标签（Key 位置）
                    yticklabels=range(seq_len),  # y 轴刻度标签（Query 位置）
                    cbar=True,                   # 显示颜色条
                    cbar_kws={"label": "attention weight"},
                )

                ax.set_xlabel("Key position (j)")
                ax.set_ylabel("Query position (i)")
                ax.set_title(f"Head {h} (batch {b})")

        plt.tight_layout()

        # 触发一次绘制：square=True 的等比例约束会把热力图收缩成正方形，
        # 但颜色条还是按收缩前的轴框占满高度，所以这里把颜色条高度对齐到热力图的真实高度
        fig.canvas.draw()
        for ax in axes.flat:
            cax = ax.collections[0].colorbar.ax
            cax_box = cax.get_position()
            ax_box = ax.get_position()
            cax.set_position([cax_box.x0, ax_box.y0, cax_box.width, ax_box.height])

        # show/savefig 放在 with 内部，确保渲染时样式仍然生效
        plt.savefig("./SA/attention_weights_multi_head.png", dpi=300)
        plt.show()
