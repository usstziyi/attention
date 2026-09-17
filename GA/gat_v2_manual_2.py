import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, leaky_relu_slope=0.2):
        super().__init__()
        self.out_features = out_features
        self.W = nn.Linear(2 * in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(out_features, 1))
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, edge_index):
        N = h.shape[0]
        src, tgt = edge_index[0], edge_index[1]

        h_src = h[src]
        h_tgt = h[tgt]
        # 1.拼接
        edge_feat = torch.cat([h_tgt, h_src], dim=1) # (E, 2 * in_features)
        # (E,2*out_features)@(2*out_features, out_features)=(E, out_features)
        # 2.线性变换
        edge_feat = self.W(edge_feat)
        # 3.激活
        act = self.leaky_relu(edge_feat)
        # (E, out_features)@(out_features, 1)=(E, 1)
        # 4.投影为标量注意力分数
        e = (act @ self.a).squeeze(-1)

        e_max = torch.full((N,), float('-inf'), dtype=e.dtype, device=e.device)
        e_max = e_max.scatter_reduce(0, tgt, e, reduce='amax')

        e_exp = torch.exp(e - e_max[tgt])

        denom = torch.zeros(N, dtype=e.dtype, device=e.device)
        denom = denom.scatter_add(0, tgt, e_exp)

        alpha = e_exp / (denom[tgt] + 1e-16)

        # 消息项只用源节点的变换特征 W·h_j，对应原论文 h_i' = σ(Σ_j α_ij W h_j)，
        # W 按列拆开后 source 对应拼接向量 [h_tgt‖h_src] 的后半部分
        W_src = self.W.weight.chunk(2, dim=1)[1]
        msg = h_src @ W_src.t()

        out = torch.zeros(N, self.out_features, dtype=h.dtype, device=h.device)
        out = out.scatter_add(
            0,
            tgt.unsqueeze(1).expand(-1, self.out_features),
            msg * alpha.unsqueeze(1)
        )
        return out


class GATv2(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,
                 num_heads=4, num_layers=2, leaky_relu_slope=0.2, dropout=0.0):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ModuleList([
                GATv2Layer(in_features, hidden_features, leaky_relu_slope)
                for _ in range(num_heads)
            ])
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.ModuleList([
                    GATv2Layer(hidden_features * num_heads, hidden_features, leaky_relu_slope)
                    for _ in range(num_heads)
                ])
            )
        self.layers.append(
            nn.ModuleList([
                GATv2Layer(hidden_features * num_heads, out_features, leaky_relu_slope)
                for _ in range(num_heads)
            ])
        )

    def forward(self, h, edge_index):
        x = h
        for i, layer_heads in enumerate(self.layers):
            head_outs = [head(x, edge_index) for head in layer_heads]
            x = torch.cat(head_outs, dim=1)
            if i < self.num_layers - 1:
                x = F.elu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.mean(dim=1) if x.dim() == 3 else x
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    N, in_features, hidden, out_features = 4, 3, 8, 5
    h = torch.randn(N, in_features)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],
        [1, 2, 0, 2, 3, 0]
    ], dtype=torch.long)

    model = GATv2(in_features, hidden, out_features, num_heads=2, num_layers=2)
    with torch.no_grad():
        out = model(h, edge_index)
    print("输入:", h.shape)
    print("输出:", out.shape)
    print(out)