import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features

        # 第一步：线性变换矩阵 W
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # 第二步：注意力权重向量 a
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414) # 初始化
        
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        # h: (N, in_features) 节点特征矩阵
        # adj: (N, N) 邻接矩阵（通常包含自环，即对角线为1）
        N = h.size(0)
        
        # 1. 线性变换 z = Wh
        Wh = self.W(h) # shape: (N, out_features)
        
        # 2. 准备拼接 (利用广播机制构造所有节点对 [z_i || z_j])
        # 这里为了避免生成极其庞大的 (N, N, 2F) 中间张量导致显存爆炸，
        # 我们通常采用一种更省显存的“分步计算”技巧：
        # e_ij = a1 * z_i + a2 * z_j，其中 a = [a1 || a2]
        
        # 将 a 拆分为两半，分别对应 z_i 和 z_j
        a1 = self.a[:self.out_features, :] # shape: (out_features, 1)
        a2 = self.a[self.out_features:, :] # shape: (out_features, 1)
        
        # 计算 a1 * z_i (对所有节点): (N, out_features) @ (out_features, 1) -> (N, 1)
        # 再转置成 (1, N)，方便后续广播
        a1_Wh = torch.matmul(Wh, a1).transpose(0, 1) # shape: (1, N)
        
        # 计算 a2 * z_j (对所有节点): (N, out_features) @ (out_features, 1) -> (N, 1)
        a2_Wh = torch.matmul(Wh, a2) # shape: (N, 1)
        
        # 利用广播机制相加，直接得到所有 (i, j) 对的分数矩阵
        # (1, N) + (N, 1) -> (N, N)
        e = self.leakyrelu(a1_Wh + a2_Wh) # shape: (N, N)
        
        # ================= 步骤 3: 归一化注意力系数 (Softmax + Mask) =================
        # 将邻接矩阵中为 0 的位置（即没有边的节点对）设置为一个极大的负数
        # 这样在 Softmax 后，这些位置的注意力系数就会变成 0
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # shape: (N, N)
        
        # 对每一行（即节点 i 的所有邻居）做 Softmax 归一化
        attention = F.softmax(attention, dim=1) # shape: (N, N), 每行和为 1
        
        # 可选：加上 Dropout 防止过拟合（GAT 论文中使用了 Dropout）
        # attention = F.dropout(attention, 0.2, training=self.training)
        
        # ================= 步骤 4: 聚合邻居特征 =================
        # 用注意力系数加权求和邻居特征: h_i' = sum_j (alpha_ij * z_j)
        # 矩阵乘法: (N, N) @ (N, out_features) -> (N, out_features)
        h_prime = torch.matmul(attention, Wh)
        
        # 经过激活函数 (GAT 论文中通常用 ELU，也可以用 ReLU)
        h_prime = F.elu(h_prime)
        
        return h_prime


# ================= 测试 Demo =================
if __name__ == "__main__":
    # 假设有 4 个节点，每个节点有 3 个特征
    N = 4
    in_features = 3
    out_features = 2 # 输出特征维度设为 2

    # 随机初始化节点特征
    h = torch.randn(N, in_features)
    print("输入节点特征 h:\n", h)

    # 构造邻接矩阵 (包含自环，即对角线为 1)
    # 0-1, 1-2, 2-3 相连
    adj = torch.tensor([
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [0., 1., 1., 1.],
        [0., 0., 1., 1.]
    ])
    print("\n邻接矩阵 adj:\n", adj)

    # 实例化 GAT 层
    gat_layer = GraphAttentionLayer(in_features, out_features)
    
    # 前向传播
    out = gat_layer(h, adj)
    print("\n输出节点特征 h':\n", out)
    print("输出形状:", out.shape) # 应该是 (4, 2)