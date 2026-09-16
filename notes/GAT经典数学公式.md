GAT 单层、单头情况下最经典的公式。

### 1. 线性变换

对每个节点特征做一次共享线性变换：

$$
z_i = W h_i
$$

### 2. 计算注意力分数

计算节点 $i$ 和邻居 $j$ 之间的未归一化注意力分数：

$$
e_{ij} = \text{LeakyReLU}\left(a^T [z_i \Vert z_j]\right)
$$

其中 $[z_i \Vert z_j]$ 表示拼接，$a$ 是可学习的权重向量。

### 3. 归一化注意力系数

对节点 $i$ 的所有邻居做 Softmax：

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$

### 4. 聚合邻居特征

用注意力系数加权求和，再经过激活函数：

$$
h_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} z_j\right)
$$

***

如果只看**最核心的一句话**，就是：

$$
h_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)
$$

其中：

$$
\alpha_{ij} = \text{softmax}_j\left(\text{LeakyReLU}\left(a^T [W h_i \Vert W h_j]\right)\right)
$$

这就是 GAT 单层、单头情况下最经典的公式。


把 GAT 单层、单头情况下的两个公式合并写，就是把 $\alpha_{ij}$ 的表达式直接代入 $h_i'$ 中，得到：

$$
h_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \text{softmax}_j\Big( \text{LeakyReLU}\left( a^T [W h_i \Vert W h_j] \right) \Big) W h_j \right)
$$

或者更明确地把 softmax 展开：

$$
h_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \frac{\exp\left( \text{LeakyReLU}\left( a^T [W h_i \Vert W h_j] \right) \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \text{LeakyReLU}\left( a^T [W h_i \Vert W h_k] \right) \right)} W h_j \right)
$$

---

### 说明

- 外层 $\sigma$：最终激活函数
- 求和 $\sum_{j \in \mathcal{N}(i)}$：只对邻居聚合
- softmax 分母 $\sum_{k \in \mathcal{N}(i)}$：在邻居范围内归一化，保证 $\sum_j \alpha_{ij} = 1$
- $a^T [W h_i \Vert W h_j]$：拼接后打分
- $W h_j$：邻居特征经过同一个线性变换，相当于 attention 里的 Value

这样写出来的好处是**一个公式完整表达 GAT 单层单头的全部计算**，不需要再分两步看。