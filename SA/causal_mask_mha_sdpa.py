import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
SDPA（Self-Attention with Positional Embedding） 缩放点积注意力掩码: True表示保留，False表示屏蔽
"""


def main():
    # diagonal=0：保留主对角线及其以下元素
    mask_0 = torch.tril(torch.ones(5, 5, dtype=torch.bool), diagonal=0)

    # diagonal=1：保留主对角线上方第一条对角线及其以下
    mask_1 = torch.tril(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    # diagonal=-1：保留主对角线下方第一条对角线及其以下（不含主对角线）
    mask_neg1 = torch.tril(torch.ones(5, 5, dtype=torch.bool), diagonal=-1)

    # 保留主对角线及上下各 1 条对角线（三对角带状掩码）
    mask_tri = torch.ones(5, 5, dtype=torch.bool)
    mask_tri = torch.tril(mask_tri, diagonal=-1) & torch.tril(mask_tri, diagonal=1)

    masks = [
        ("diagonal=0", mask_0),
        ("diagonal=1", mask_1),
        ("diagonal=-1", mask_neg1),
        ("diagonal=±1,0", mask_tri),
    ]

    for name, m in masks:
        print(f"{name}:")
        print(m)
        print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # 自定义颜色：0 显示绿色，1 显示白色（不显示）
    cmap_01 = ListedColormap(["#f3f3f3", "#0FDC78"])
    for ax, (name, m) in zip(axes.ravel(), masks):
        sns.heatmap(
            m.int().numpy(),   # 数据：bool 掩码转 int(0/1) 并转 numpy 数组
            ax=ax,             # 绘制到哪个子图
            cbar=False,        # 不显示右侧颜色条
            annot=True,        # 在热力图中显示数值
            fmt="d",           # 标注格式：整数
            cmap=cmap_01,      # 颜色映射：0 有颜色，1 白色不显示
            xticklabels=True,  # 显示 x 轴刻度标签
            yticklabels=True   # 显示 y 轴刻度标签
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()