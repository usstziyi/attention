import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# ================= 1. 构造图数据 =================
# 使用 PyG 的 Data 对象来管理图数据
# 节点特征: 4 个节点，每个节点 3 维特征
x = torch.randn(4, 3)

# 边列表: 和之前一样 (源节点 -> 目标节点)
edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 3],  # 源节点
    [1, 2, 0, 2, 3, 0]   # 目标节点
], dtype=torch.long)

# 封装成 PyG 的 Data 对象
data = Data(x=x, edge_index=edge_index)

print(f"节点特征维度: {data.x.shape}")
print(f"边数量: {data.edge_index.shape[1]}")

# ================= 2. 定义 GAT 模型 =================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # 第一层 是 特征提取层 ， 第二层 是 输出映射层 ：
        # 第一层 GAT: 多头注意力 (输出维度 = hidden_channels * heads)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # 第二层 GAT: 单头注意力，concat=False 表示多头结果取平均而不是拼接
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, 
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # 第一层 + ELU 激活
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        # 第二层
        x = self.conv2(x, edge_index)
        return x

# ================= 3. 前向传播 =================
model = GAT(in_channels=3, hidden_channels=8, out_channels=5, heads=4)
model.eval()  # 关闭 dropout，方便查看结果

with torch.no_grad():
    output = model(data.x, data.edge_index)

print(f"\n输出节点特征维度: {output.shape}")
print("输出所有节点特征:")
print(output)
