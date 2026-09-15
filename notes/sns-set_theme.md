# `sns.set_theme(style="white", context="notebook", font_scale=1.0)` 解释

代码位置：`SA/sa.py` 第 84 行

```python
    sns.set_theme(style="white", context="notebook", font_scale=1.0)
```

`sns.set_theme(...)` 是 seaborn 的**全局样式设置**，一次性调好之后所有 matplotlib/seaborn 图都受影响。

## 三个参数

- `style="white"`：控制**背景和网格线**。`white` 表示纯白背景、无网格线。其他可选 `darkgrid`（seaborn 默认，灰底白网格）、`whitegrid`、`dark`、`ticks`。热力图本身就是色块，加网格反而干扰，所以用 `white`。
- `context="notebook"`：控制**元素的整体尺寸**（字体、线条粗细、刻度大小等一套预设）。可选 `paper` < `notebook` < `talk` < `poster`，越往后字越大。`notebook` 是默认值，适合屏幕查看。
- `font_scale=1.0`：在 `context` 基础上**再整体缩放字体**，1.0 表示不缩放。想让字大一点就调成 1.2、1.5。

## 等价写法

它约等于下面三件事的合集：

```python
sns.set_style("white")        # 背景、网格
sns.set_context("notebook", font_scale=1.0)   # 元素尺寸与字号
sns.set_palette("deep")       # 默认调色板（此处未指定，用 seaborn 默认色板）
```

## 注意点

- **它是全局副作用**，会修改 matplotlib 的 rcParams，不只是影响紧接着的那张图。如果同一进程里还要画别的图、或者这个文件被 import 使用，样式会被一起改掉。放在 `if __name__ == "__main__":` 内，影响可控。
- `color=` 之类的显式参数优先级高于主题，比如 heatmap 里的 `cmap="GnBu"` 不受主题调色板影响。
- 返回的是被修改的 rcParams 字典，一般不用管。

如果只想让这张图生效而不污染全局，可以用上下文管理器：

```python
with sns.axes_style("white"), sns.plotting_context("notebook", font_scale=1.0):
    ...
```

## 补充

`SA/sa.py` 第 84 行其实可以省略——`style="white"` 和 `font_scale=1.0` 都是常用默认值附近的选择，不写也能跑，只是背景会变成 seaborn 默认的浅灰网格风格。
