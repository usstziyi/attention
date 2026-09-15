"""遍历并可视化 seaborn 能用的所有 colormap。

seaborn 自身只提供 6 个 colormap（rocket/mako/flare/crest/icefire/vlag），
其余全部透传自 matplotlib，因此这里直接枚举 matplotlib 的 colormap 注册表
（import seaborn 时它会把自有的 6 个注册进去，所以能一并列出来）。

任意名字加 "_r" 后缀即为反向版本，如 "viridis_r"、"RdBu_r"。
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401  导入时会注册 seaborn 自有的 colormap

# seaborn 自有的 colormap，枚举时用 ★ 标出
SEABORN_CMAPS = {"rocket", "mako", "flare", "crest", "icefire", "vlag"}

GRADIENT = np.linspace(0, 1, 256).reshape(1, -1)  # 一行渐变，用来展示 colormap
COLS = 4        # 每行放几个色条
ROWS_PER_FIG = 5  # 每张图放几行


def draw_page(names, page_idx, total_pages):
    """把一组 colormap 画成网格，每个格子里是 0~1 的渐变色条。"""
    rows = math.ceil(len(names) / COLS)
    fig, axes = plt.subplots(rows, COLS, figsize=(12, 1.1 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        ax.imshow(GRADIENT, aspect="auto", cmap=name)
        ax.set_title(name + ("  ★" if name in SEABORN_CMAPS else ""), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(names):]:  # 用完的格子留空
        ax.axis("off")

    fig.suptitle(f"colormaps ({page_idx}/{total_pages})", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    names = sorted(plt.colormaps())
    per_page = COLS * ROWS_PER_FIG
    pages = [names[i:i + per_page] for i in range(0, len(names), per_page)]

    print(f"共 {len(names)} 个 colormap，其中 seaborn 自有：")
    print("  ", sorted(SEABORN_CMAPS & set(names)))
    print(f"分 {len(pages)} 张图展示，★ 表示 seaborn 自有；加 _r 后缀即反向版本")

    for i, page in enumerate(pages, 1):
        draw_page(page, i, len(pages))

    plt.show()


if __name__ == "__main__":
    main()
