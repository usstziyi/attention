"""用 PyTorch 内置的 nn.MultiheadAttention 实现多头自注意力 (Self-Attention)。

与 sa_sdpa.py 的关系:
- sa_sdpa.py 自己声明 W_q/W_k/W_v, 再用 F.scaled_dot_product_attention 算注意力;
- 本文件把投影、分头、注意力、合并、输出投影整条链路都交给 nn.MultiheadAttention,
  它内部把 W_q/W_k/W_v 拼成一个 in_proj_weight (切分后按 q/k/v 顺序排列),
  多头就是把 d_model 切成 num_heads 份, 每份独立算一次缩放点积注意力再拼回来,
  最后过一层 out_proj 得到输出。

几点与手写版的差异 (容易踩坑):
1. 输入默认是 (T, B, d_model), 本文用 batch_first=True 改成 (B, T, d_model);
2. 自注意力时 query/key/value 传同一个张量;
3. 因果屏蔽不能只给 is_causal=True: 内部要求同时给出 attn_mask 作为提示,
   且 need_weights=True 时 is_causal 没实现 (见 torch/nn/functional.py)。
   所以这里自己按 (T, T) 生成下三角 mask 一并传入, 语义与 SDPA 正好相反
   —— bool 张量里 True 表示"屏蔽";
4. need_weights=True 才会返回注意力权重, 但它会强制走非 flash 的慢路径, 也拿不到梯度优化的好处,
   所以只在需要观察权重时开启;
5. 返回的权重默认已对多头做平均, average_attn_weights=False 才拿到 (B, H, T, T) 的逐头权重;
6. d_model 必须能被 num_heads 整除, 且不像 SDPA 那样可以自由指定 d_k/d_v。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionMHA(nn.Module):
    """
    多头自注意力 (Self-Attention), 底层调用 nn.MultiheadAttention
    输入 x: (batch_size, seq_len, d_model)
    输出:   (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # W_q/W_k/W_v 与分头逻辑都在这里, batch_first=True 让输入按 (B, T, d_model) 排布
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

    def forward(self, x, is_causal=False, need_attn=False, average_attn_weights=True):
        # x: (B, T, d_model); 自注意力: q = k = v = x
        attn_mask = None
        if is_causal:
            # 自注意力下 L == S, 直接生成 (T, T) 下三角 mask
            # 注意约定: nn.MultiheadAttention 里 bool mask 的 True 表示"不允许关注"
            T = x.size(1)
            attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )

        # need_weights=False 时 is_causal 会直接提示给 SDPA, 走 flash 等快速后端
        out, attn = self.mha(
            x, x, x,
            attn_mask=attn_mask,
            is_causal=is_causal,
            need_weights=need_attn,
            average_attn_weights=average_attn_weights,
        )  # out: (B, T, d_model); attn: (B, T, T) 或 (B, H, T, T)

        return out, attn


# ================== 测试示例 ==================
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)  # 模拟 EEG 电极通道特征

    # 因果掩码（causal mask）：位置 i 只能关注 j <= i，未来位置 j > i 被屏蔽
    # 依旧是 is_causal=True 的用法，mask 由 SelfAttentionMHA 内部按 MHA 的约定生成
    attn_layer = SelfAttentionMHA(d_model, num_heads)
    out, attn_weights = attn_layer(x, is_causal=True, need_attn=True)

    print("输入形状: ", x.shape)               # (2, 5, 8)
    print("输出形状: ", out.shape)             # (2, 5, 8)
    print("注意力权重形状:", attn_weights.shape) # (2, 5, 5)，已对 2 个头求平均

    # 检查注意力权重每一行是否归一化（和为 1）
    print("权重行和: ", attn_weights.sum(dim=-1))

    # 检查因果掩码是否生效：上三角（未来位置）权重应全为 0
    upper = torch.triu(attn_weights, diagonal=1)
    print("上三角(未来位置)最大权重:", upper.abs().max().item())

    # 一致性检查：用逐头权重 @ V 手工重算，结果应等于模块输出
    _, attn_per_head = attn_layer(x, is_causal=True, need_attn=True, average_attn_weights=False)
    print("逐头权重形状:", attn_per_head.shape)  # (2, 2, 5, 5)

    # 模块把 W_q/W_k/W_v 拼成了一个 in_proj_weight，拆出来取 W_v 对应的那一段
    W_v = attn_layer.mha.in_proj_weight.chunk(3, dim=0)[2]
    b_v = attn_layer.mha.in_proj_bias.chunk(3, dim=0)[2]
    V = F.linear(x, W_v, b_v)  # (B, T, d_model)
    V = V.reshape(batch_size, seq_len, num_heads, attn_layer.head_dim).transpose(1, 2)

    ref = attn_per_head @ V                                        # (B, H, T, head_dim)
    ref = ref.transpose(1, 2).reshape(batch_size, seq_len, d_model)  # (B, T, d_model)
    ref = attn_layer.mha.out_proj(ref)
    print("与 逐头权重@V 的最大误差:", (out - ref).abs().max().item())

    # ================== 可视化注意力权重 (seaborn) ==================
    # attn_weights: (B, T, T)，第 i 行第 j 列表示位置 i 对位置 j 的关注程度（多头平均后）
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
        plt.savefig("./SA/attention_weights_mha.png", dpi=300)
        plt.show()
