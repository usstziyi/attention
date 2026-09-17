import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # 第一层 是 特征提取层 ， 第二层 是 输出映射层 ：
        # 第一层 GAT: 多头注意力 (输出特征维度 = hidden_channels * heads)
        # 输出特征维度是`hidden_channels * heads` = 8 × 4 = 32 ，不是 8。因为`GATConv` 默认`concat=True` ，会把 4 个头的结果 拼接 在一起。
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # 第二层 GAT: 单头注意力，concat=False 表示多头结果取平均而不是拼接
        # 输出特征维度是`out_channels` = 5。
        # 因为`GATConv` 默认`concat=False` ，会把 1 个头的结果 取平均。
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层 + ELU 激活
        # x: (4, 3)
        # edge_index: (2, 6)
        # output: (4, 32)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        # 第二层
        # x: (4, 32)
        # edge_index: (2, 6)
        # output: (4, 5)
        x = self.conv2(x, edge_index)
        return x # (4, 5)



def main():
    # 节点
    x = torch.randn(4, 3)

    # 边列表: (源节点 -> 目标节点)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],  # 源节点
        [1, 2, 0, 2, 3, 0]   # 目标节点
    ], dtype=torch.long)

    # 封装成 PyG 的 Data 对象
    data = Data(x=x, edge_index=edge_index)


    model = GAT(in_channels=3, hidden_channels=8, out_channels=5, heads=4)
    model.eval()  # 关闭 dropout，方便查看结果

    with torch.no_grad():

        # data.x: (4, 3)
        # data.edge_index: (2, 6)
        # output: (4, 5)
        output = model(data)

    print("输入节点特征:")
    print(x)
    print("输出所有节点特征:")
    print(output)

if __name__ == "__main__":
    main()
