import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
MHA（Multi-Head Attention） 多头自注意力掩码: False表示可看，True表示不可看
"""


def main():
    # 此保留是矩阵本身的概念，不是因果掩码的保留
    # j>=i+1;保留主对角线右边第一条对角线及其以上（不含主对角线）,其余置为False
    mask_1 = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    # j>=i+0;保留主对角线及其以上元素,其余置为False
    mask_0 = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=0)

    # j>=i-1;保留主对角线左方第一条对角线及其以上,其余置为False
    mask_neg1 = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=-1)

    # 全True矩阵
    mask_tri = torch.ones(5, 5, dtype=torch.bool)
    # j>=i-2，保留主对角线左方第二条对角线及其以上,其余置为False
    mask_tri = torch.triu(mask_tri, diagonal=-2)
    # j<=i+0，保留主对角线及其以下元素,其余置为False
    mask_tri = torch.tril(mask_tri, diagonal=0)
    # 取反
    mask_tri = ~mask_tri



    masks = [
        ("diagonal=1", mask_1),
        ("diagonal=0", mask_0),
        ("diagonal=-1", mask_neg1),
        ("diagonal=-2,0", mask_tri),
    ]

    for name, m in masks:
        print(f"{name}:")
        print(m)
        print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # 自定义颜色：0 显示绿色，1 显示白色（不显示）
    cmap_01 = ListedColormap(["#0FDC78", "#f3f3f3"])
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