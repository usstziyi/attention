`torch_scatter` 是 PyTorch 的扩展库，专门用于处理稀疏数据和图神经网络中的“散布-聚合”（scatter-reduce）操作。它解决的核心问题是：当你有大量元素（比如图中的边或节点特征），需要按某种索引将它们分组并聚合时，PyTorch 原生操作不够高效或不够直观。这个库提供了高度优化的 GPU 实现，在 GNN 中几乎是标配。

### 核心概念：Scatter 与 Segment

- **Scatter 操作**：根据 `index` 张量，把 `src` 中的值“散布”到输出张量的对应位置，并沿指定维度做聚合（求和、取最大等）。索引不要求有序。
- **Segment 操作**：类似 scatter，但索引必须是有序的（或通过指针指定区间），通常速度更快且完全确定性。

### 常用函数速查

| 函数 | 返回值 | 说明 |
|------|--------|------|
| `scatter_sum(src, index, dim)` | Tensor | 求和，也常写作 `scatter_add` |
| `scatter_max(src, index, dim)` | (values, argmax) | 取最大值及对应索引位置 |
| `scatter_min(src, index, dim)` | (values, argmin) | 取最小值及对应索引位置 |
| `scatter_mean(src, index, dim)` | Tensor | 求均值 |
| `scatter_softmax(...)` | Tensor | 在分组内做 softmax |
| `segment_csr(src, indptr)` | Tensor | 按指针数组分段聚合，速度最快 |



### 总结

`torch_scatter` 的本质是**按索引分组聚合**。`scatter_sum` 和 `scatter_max` 在 GNN 的消息传递、节点池化中极为常用。使用时注意三点：索引可以是任意顺序、空组需要用 `dim_size` 显式保留、max/min 的空组填充值用 `fill_value` 控制。