# torch.scatter_reduce_ 与 torch.gather

整理自 DeepSeek 对话：https://chat.deepseek.com/share/ucunj05kwki9syg8cd

---

## 一、scatter_reduce_ 是什么

`torch.scatter_reduce_` 是 PyTorch 原生提供的**按索引分组归约**操作，可以理解为 `scatter_add` 的"升级版"——后者只能做加法，而它能做多种归约。

把 `src` 里的值按照 `index` 指定的位置"散布"到输出张量中，同一位置有多个值时按指定方式（求和、取最大等）归约。

### 函数签名

```python
Tensor.scatter_reduce_(dim, index, src, reduce, *, include_self=True)
```

| 参数 | 含义 |
|------|------|
| `dim` | 沿哪个维度散布 |
| `index` | 索引张量，形状需与 `src` 匹配 |
| `src` | 源数据 |
| `reduce` | 归约方式：`"sum"` / `"prod"` / `"mean"` / `"amax"` / `"amin"` |
| `include_self` | 是否把输出张量原来的值也纳入归约（默认 `True`） |

### 例子

```python
import torch

src = torch.tensor([2.0, 3.0, 1.0, 4.0, 5.0])
index = torch.tensor([0, 0, 1, 2, 2])

# 按组求和
out = torch.zeros(3)
out.scatter_reduce_(0, index, src, reduce="sum")
print(out)  # tensor([5., 1., 9.])

# 按组取最大
out = torch.zeros(3)
out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
print(out)  # tensor([3., 1., 5.])
```

### 关键点

- **带下划线 `_` 表示原地操作**：直接修改调用它的张量并返回该张量本身。不带下划线的 `torch.scatter_reduce` 返回新张量。
- **`include_self` 很重要**：输出张量初始为 0 时，做 `"sum"` 设 `include_self=True` 没问题；但做 `"amax"` 时 0 会参与比较，可能污染结果。取最大/最小时通常设 `include_self=False`，或把初始值设为 `-inf`。
- **和 `torch_scatter` 的关系**：PyTorch 引入这个函数就是为了对齐 `torch_scatter` 的核心功能集和性能，PyG 2.3+ 内部已经用它替代 `torch_scatter`。
- **和 `scatter_add_` 的区别**：`scatter_add_` 是 `scatter_reduce_` 在 `reduce="sum"` 时的特例，功能更窄，旧代码中仍有使用。

---

## 二、`dim` 到底指定的是谁

`dim` 指定的是**输出张量 `out` 的维度**，准确说：`index` 里的数值，对应的是 `out` 在 `dim` 这个轴上的下标。

因为 `scatter_reduce_` 是原地操作，它直接修改调用它的张量（即 `out` 本身）。

### 一维例子

```python
src = torch.tensor([2.0, 3.0, 1.0, 4.0, 5.0])
index = torch.tensor([0, 0, 1, 2, 2])
out = torch.zeros(3)

out.scatter_reduce_(0, index, src, reduce="sum")
# tensor([5., 1., 9.])
```

`dim=0`，`index` 里的 `0, 0, 1, 2, 2` 就是 `out` 第 0 维（唯一那一维）上的下标。`src[0]=2` 和 `src[1]=3` 都指向位置 0，归约成 `5`。

### 二维例子（`dim` 的作用才明显）

```python
src = torch.tensor([[1., 2.],
                    [3., 4.],
                    [5., 6.]])
index = torch.tensor([[0, 0],
                      [1, 1],
                      [0, 0]])
out = torch.zeros(2, 2)

out.scatter_reduce_(0, index, src, reduce="sum")
# tensor([[6., 8.],
#         [3., 4.]])
```

这里 `dim=0`，`index` 指定的是**行下标**：

- `src[0] = [1, 2]`，`index[0] = [0, 0]` → 放到第 0 行
- `src[1] = [3, 4]`，`index[1] = [1, 1]` → 放到第 1 行
- `src[2] = [5, 6]`，`index[2] = [0, 0]` → 也放到第 0 行

第 0 行有两组值 `[1,2]` 和 `[5,6]`，求和得 `[6, 8]`。**列坐标没有被 `index` 指定**，它按 `src` 自身的位置自然对齐。

如果改成 `dim=1`：

```python
out = torch.zeros(3, 2)
out.scatter_reduce_(1, index, src, reduce="sum")
```

这时 `index` 指定的是**列下标**，`src` 的每一行会按 `index` 对应行里的列坐标散布到输出行的对应列。

### 三者的关系

- `index` 的数值 → 指向 `out` 在 `dim` 轴上的位置
- `src` 的数值 → 被散射过去、参与归约的数据
- `index` 与 `src` **形状相同、逐元素一一对应**

**一句话**：`dim` 指定的是 `out` 的轴，`index` 里的数字是 `out` 在该轴上的下标，其余维度按元素在 `src` 中的自然位置对齐。

---

## 三、逐元素对应规则（以 `dim=0` 为例）

`index` 与 `src` 形状相同，逐元素一一对应。对每个位置 `(i, j)` 上的 `src[i][j]`：

- **列坐标 `j` 保持不变**（列不是 `dim`，不参与索引）
- **行坐标由 `index[i][j]` 决定**（行是 `dim=0`，由 index 指定）

所以 `index` 里每个数字都能独立取值，同一行的两个数字**没有必须相同的约束**。

### `index` 每行取值不同

```python
src = torch.tensor([[1., 2.],
                    [3., 4.],
                    [5., 6.]])
index = torch.tensor([[0, 1],
                      [1, 0],
                      [0, 0]])
out = torch.zeros(2, 2)
out.scatter_reduce_(0, index, src, reduce="sum")
```

`dim=0`，`index` 里的数字是 `out` 的行下标。逐行看：

**第 0 行**（`src[0] = [1, 2]`，`index[0] = [0, 1]`）

- `src[0][0] = 1` → 列 0，`index[0][0] = 0` → 放到 `out[0, 0]`
- `src[0][1] = 2` → 列 1，`index[0][1] = 1` → 放到 `out[1, 1]`

**第 1 行**（`src[1] = [3, 4]`，`index[1] = [1, 0]`）

- `src[1][0] = 3` → 列 0，`index[1][0] = 1` → 放到 `out[1, 0]`
- `src[1][1] = 4` → 列 1，`index[1][1] = 0` → 放到 `out[0, 1]`

**第 2 行**（`src[2] = [5, 6]`，`index[2] = [0, 0]`）

- `src[2][0] = 5` → 列 0，`index[2][0] = 0` → 放到 `out[0, 0]`
- `src[2][1] = 6` → 列 1，`index[2][1] = 0` → 放到 `out[0, 1]`

汇总：

```text
out[0, 0] = 1 + 5 = 6
out[0, 1] = 4 + 6 = 10
out[1, 0] = 3
out[1, 1] = 2
```

结果：

```text
tensor([[ 6., 10.],
        [ 3.,  2.]])
```

---

## 四、scatter_reduce_ 的"逆操作"

**没有严格的单一逆操作函数**，因为逆向过程取决于归约方式（`sum`、`prod`、`amax` 等）。

### 最接近的概念：gather

- `scatter` 是**分散**：根据 `index` 把值"写"到目标位置
- `gather` 是**聚集**：根据 `index` 把值从源位置"取"出来

在 PyTorch 的 autograd 实现中，`scatter_reduce_` 反向传播对 `sum` 归约直接调用的就是 `grad.gather(dim, index)`。这说明在"梯度流动"的意义上，`gather` 承担了 `scatter_reduce_` 反向的"逆操作"角色。

### 为什么没有真正的"逆"

因为 `scatter_reduce_` 是**多对一**操作：多个 `src` 元素可能映射到同一个输出位置并被归约，原始信息被合并或丢弃，无法从输出唯一恢复输入。

| 归约模式 | 逆向操作思路 |
|---|---|
| `"sum"` | 理论上不可逆（只知道总和，不知道各个加数）；梯度反向传播中直接用 `gather` 按索引取值 |
| `"amax"` / `"amin"` | 反向传播时梯度只流向确实取到最大值/最小值的 `src` 位置 |
| `"prod"` | 反向涉及除以原值等操作，且有零值特殊处理 |

结论：问"从输出恢复输入"的函数 → **没有**；问"梯度反向中配合使用的操作" → **`torch.gather`**。

---

## 五、torch.gather 的用法

`torch.gather` 是**按索引取值**的操作，可理解为 `scatter` 的逆向：`scatter` 是"按索引写"，`gather` 是"按索引读"。

### 函数签名

```python
torch.gather(input, dim, index)   # 或 input.gather(dim, index)
```

- `input`：源张量，从这里取值
- `dim`：沿哪个维度按 `index` 取
- `index`：索引张量，**形状决定输出形状**

关键规则：`index` 的形状决定输出形状；`index` 里的值是 `input` 在 `dim` 轴上的下标；其余维度按元素在 `index` 中的自然位置对齐。

### 一维例子

```python
import torch

input = torch.tensor([10., 20., 30., 40.])
index = torch.tensor([0, 2, 3])

out = torch.gather(input, 0, index)
print(out)  # tensor([10., 30., 40.])
```

### 二维例子（`dim=0`，按行取）

```python
input = torch.tensor([[1., 2.],
                      [3., 4.],
                      [5., 6.]])
index = torch.tensor([[0, 1],
                      [2, 0]])

out = torch.gather(input, 0, index)
print(out)
# tensor([[1., 4.],
#         [5., 2.]])
```

`dim=0`，`index` 里的值是**行下标**，列坐标保持不变：

- `out[0][0]`：`index[0][0]=0` → `input[0][0] = 1`
- `out[0][1]`：`index[0][1]=1` → `input[1][1] = 4`
- `out[1][0]`：`index[1][0]=2` → `input[2][0] = 5`
- `out[1][1]`：`index[1][1]=0` → `input[0][1] = 2`

### 二维例子（`dim=1`，按列取）

```python
input = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]])
index = torch.tensor([[2, 0],
                      [1, 2]])

out = torch.gather(input, 1, index)
print(out)
# tensor([[3., 1.],
#         [5., 6.]])
```

`dim=1`，`index` 里的值是**列下标**，行坐标保持不变：

- `out[0][0]`：`index[0][0]=2` → `input[0][2] = 3`
- `out[0][1]`：`index[0][1]=0` → `input[0][0] = 1`
- `out[1][0]`：`index[1][0]=1` → `input[1][1] = 5`
- `out[1][1]`：`index[1][1]=2` → `input[1][2] = 6`

### 与 scatter 的对称关系

```python
# scatter：按 index 把 src 写到 out
out.scatter_reduce_(0, index, src, reduce="sum")

# gather：按 index 从 input 读到 out
out = torch.gather(input, 0, index)
```

用同一个 `index` 时方向正好相反：`scatter` 是 `index → out 位置`，`gather` 是 `index → input 位置`。

### 两个易错点

1. **`index` 的值必须在合法范围内**：`dim=0` 时 `index` 里的值必须在 `[0, input.size(0))` 之间，否则报错。
2. **`index` 的形状决定输出形状**，不是 `input` 的形状。这一点与 scatter 里"`index` 与 `src` 同形状"的规则正好对应。

### 常见用途：从 logits 中按标签取分数

```python
logits = torch.tensor([[0.1, 0.9, 0.3],
                       [0.7, 0.2, 0.8]])
labels = torch.tensor([1, 2])  # 每个样本的真实类别

# 取出每个样本对应真实类别的分数
scores = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
print(scores)  # tensor([0.9000, 0.8000])
```

这是 `gather` 在分类任务里最典型的用法：从每行 logits 中按标签列下标取出对应的预测分数。

---

## 六、专有名词澄清："列坐标保持不变" / "index 里的值是行下标"

### 1. "列坐标保持不变"是什么意思

“列坐标不变”的**出发点（基准）是输出张量 `out` 当前正在计算的位置**。

具体来说，规则是：**`out` 里当前元素在第几列，就去 `input` 的同一列取值。**

### 🔍 结合例子看（以 `out[1][0]` 为例）

当我们要计算 `out[1][0]` 这个位置时：

1.  **看 `out` 的位置**：它是第 **1 行**、第 **0 列**。
2.  **看 `index` 的值**：查 `index[1][0]`，值是 `2`。这决定了我们要去 `input` 的**第 2 行**拿数据。
3.  **应用“列坐标不变”**：因为 `out[1][0]` 在第 0 列，所以去 `input` 里也要拿**第 0 列**的数据。
4.  **得出结果**：最终取到的就是 `input` 的第 2 行、第 0 列，即 `input[2][0] = 5`。

### 🔑 总结

*   **行坐标由谁定？** 由 `index` 里的数值定。
*   **列坐标由谁定？** 由当前在 `out` 里的位置定，**不是由 `index` 定**。
*   所以，出发点是 `out` 的当前位置，以它自己的列坐标为准，去 `input` 里找同一列。

### 2. "index 里的值是行下标"是什么意思

“`index` 里的值是行下标”意思是：在这个 `dim=0` 的场景下，**`index` 张量里填写的每一个数字，都代表要在 `input` 中取哪一行。**

你可以把 `index` 当成一个“**行号清单**”，它明确告诉你输出里每个位置该去 `input` 的哪一行拿数据。

### 🔍 结合图里的例子拆解

你的 `input` 有 3 行：
- 第 0 行：`[1., 2.]`
- 第 1 行：`[3., 4.]`
- 第 2 行：`[5., 6.]`

`index` 里的数字分别是 `0, 1, 2, 0`，它们全都是**合法的行号**。我们逐个看：

*   **取 `out[0][0]` 时**：看 `index[0][0]`，值是 **0**。这表示“去 `input` 的**第 0 行**拿数据”。列坐标不变（这里是第 0 列），所以拿到 `input[0][0] = 1`。
*   **取 `out[0][1]` 时**：看 `index[0][1]`，值是 **1**。这表示“去 `input` 的**第 1 行**拿数据”。列坐标不变（这里是第 1 列），所以拿到 `input[1][1] = 4`。
*   **取 `out[1][0]` 时**：看 `index[1][0]`，值是 **2**。这表示“去 `input` 的**第 2 行**拿数据”。列坐标不变（这里是第 0 列），所以拿到 `input[2][0] = 5`。
*   **取 `out[1][1]` 时**：看 `index[1][1]`，值是 **0**。这表示“去 `input` 的**第 0 行**拿数据”。列坐标不变（这里是第 1 列），所以拿到 `input[0][1] = 2`。

### 🔑 为什么叫“行下标”？

因为 `dim=0` 指定的维度就是“行”这个维度。`index` 里的数字 `0、1、2` 直接对应 `input` 的第 0、1、2 行，它们**只负责决定行号**，不负责决定列号。列号则由当前正在处理的位置自然决定。

所以“`index` 里的值是行下标”可以翻译成：**`index` 这个矩阵里装的不是数据本身，而是“去 `input` 里取第几行”的指示牌。**




