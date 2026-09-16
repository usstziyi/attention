import torch
from torch_scatter import scatter_sum, scatter_max

# 5 个节点的特征值（假设 1 维特征）
src = torch.tensor([2.0, 3.0, 1.0, 4.0, 5.0])
# 每个节点属于哪个组（0, 1, 2）
index = torch.tensor([0, 0, 1, 2, 2])

# 按组求和
out_sum = scatter_sum(src, index, dim=0)
print("Sum per group:", out_sum)
# tensor([5., 1., 9.])  → 组0: 2+3=5, 组1: 1, 组2: 4+5=9

# 按组取最大值，返回值和 argmax（在 src 中的位置）
out_max, argmax = scatter_max(src, index, dim=0)
print("Max per group:", out_max)
# tensor([3., 1., 5.])
print("Argmax positions:", argmax)
# tensor([1, 2, 4])  → 组0最大值在位置1，组2最大值在位置4