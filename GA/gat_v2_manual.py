import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadGATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, leaky_relu_slope=0.2):
        super(SingleHeadGATv2Layer, self).__init__()

        self.W = nn.Linear(2 * in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, h, edge_index):
        N = h.shape[0]
        E = edge_index.shape[1]
        src_node_index = edge_index[0]
        tgt_node_index = edge_index[1]

        h_i = h[tgt_node_index]
        h_j = h[src_node_index]

        # W[h_i∣∣h_j]
        edge_h_cat = torch.cat([h_i, h_j], dim=1)
        Wh_cat = self.W(edge_h_cat)

        e = torch.matmul(self.leakyrelu(Wh_cat), self.a).squeeze(1)

        max_e = torch.zeros(N, dtype=e.dtype, device=e.device)
        max_e = max_e.scatter_reduce(
            dim=0,
            index=tgt_node_index,
            src=e,
            reduce='amax',
            include_self=False
        )

        e_stable = e - max_e[tgt_node_index]

        exp_e = torch.exp(e_stable)

        sum_exp = torch.zeros(N, dtype=e.dtype, device=e.device)
        sum_exp = sum_exp.scatter_add(
            dim=0,
            index=tgt_node_index,
            src=exp_e
        )

        alpha = exp_e / (sum_exp[tgt_node_index] + 1e-10)

        weighted_features = h_j * alpha.unsqueeze(1)

        out_features = torch.zeros((N, h.shape[1]), dtype=h.dtype, device=h.device)
        out_features = out_features.scatter_add(
            dim=0,
            index=tgt_node_index.unsqueeze(1).expand(-1, h.shape[1]),
            src=weighted_features
        )

        return out_features


if __name__ == "__main__":
    N = 4
    in_features = 3
    out_features = 5

    h = torch.randn(N, in_features)

    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],
        [1, 2, 0, 2, 3, 0]
    ], dtype=torch.long)

    print("输入节点特征:")
    print(h)
    print(f"输入节点特征维度: {h.shape}")
    print(f"边数量: {edge_index.shape[1]}")

    single_gatv2_model = SingleHeadGATv2Layer(in_features, out_features)

    with torch.no_grad():
        output = single_gatv2_model(h, edge_index)

    print(f"\n输出节点特征维度: {output.shape}")
    print("输出所有节点特征:")
    print(output)