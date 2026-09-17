import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, add_self_loops=True):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.6, add_self_loops=add_self_loops)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6, add_self_loops=add_self_loops)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    x = torch.randn(4, 3)

    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],
        [1, 2, 0, 2, 3, 0]
    ], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index
    )

    model = GATv2(in_channels=3, hidden_channels=8, out_channels=5, heads=4, add_self_loops=False)
    model.eval()

    with torch.no_grad():
        output = model(data)

    print("输入节点特征:")
    print(x)
    print("输出所有节点特征:")
    print(output)


if __name__ == "__main__":
    main()
