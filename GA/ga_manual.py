import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, leaky_relu_slope=0.2):
        super(SingleHeadGATLayer, self).__init__()
        
        # 1. 特征变换权重 W: [in_features, out_features]
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # 2. 注意力权重向量 a: [2 * out_features, 1] (因为需要拼接两个节点的特征)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        
        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, h, edge_index):
        """
        h: 节点特征矩阵, shape: [N, in_features]
        edge_index: 边列表, shape: [2, E], 每一列是 (源节点, 目标节点)
        """
        # h(4,3)
        N = h.shape[0] # 节点数量=4
        
        # Step 1: 线性变换 (所有节点并行)
        # Wh: [N, out_features]
        # Wh(4,5)
        Wh = self.W(h)
        
        # Step 2: 边缘并行 (Edge Parallelism) - 计算注意力分数
        # 提取每条边的源节点和目标节点特征
        # edge_index[0] 是源节点索引:[0,0,1,1,2,3]
        # edge_index[1] 是目标节点索引:[1,2,0,2,3,0]
        edge_h_src = Wh[edge_index[0]]  # [E, out_features]
        edge_h_dst = Wh[edge_index[1]]  # [E, out_features]
        
        # 拼接: [E, 2 * out_features]
        edge_h_cat = torch.cat([edge_h_src, edge_h_dst], dim=1)
        """
        edge_h_cat[e] = [ src_0, src_1, src_2, src_3, src_4, dst_0, dst_1, dst_2, dst_3, dst_4 ]
                  └────────── edge_h_src ──────────┘  └────────── edge_h_dst ────────────────┘
        """
        
        # 计算未归一化的注意力分数 e_ij: [E, 1]
        # 对应公式: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        e = self.leakyrelu(torch.matmul(edge_h_cat, self.a)).squeeze(1) # [E]
        # `e` 是 每条边一个的「未归一化注意力分数」 ，形状`[E]` 。
        # 它是 softmax 之前的 原始 logit ，还不是最终的注意力权重。
        # 「源节点的特征」对「目标节点」来说有多重要 的原始打分。
        """
        e[0]  -> 对应边 0->1 的分数logit
        e[1]  -> 对应边 0->2 的分数logit
        e[2]  -> 对应边 1->0 的分数logit
        e[3]  -> 对应边 1->2 的分数logit
        e[4]  -> 对应边 2->3 的分数logit
        e[5]  -> 对应边 3->0 的分数logit
        """
        
        # Step 3: 分段 Softmax (Segment Softmax) - 按目标节点归一化
        # 我们需要对每个目标节点 i 的所有邻居的 e_ij 做 Softmax
        # 使用 scatter_reduce 来实现按目标节点分组求最大值和指数和
        
        # 减去最大值防止溢出 (数值稳定性)
        # 创建一个极小的初始张量来存储每个节点的最大分数
        # N: 节点数量, 例如4
        max_e = torch.full((N,), -1e9, dtype=e.dtype, device=e.device)
        # 遍历每条边 k，把`e[k]` 写到输出张量的`index[k]` 号槽位；
        # 如果该槽位已被写过（多条边指向同一节点），就用`amax` 保留更大的那个。
        # 输出形状是`[N]` ，即每个节点一个“最大入边分数”。
        # src:e[0] e[1] e[2] e[3] e[4] e[5]
        # dst:edge_index[1]:[1,2,0,2,3,0]
        # out:max_e[0] max_e[1] max_e[2] max_e[3]
        max_e = max_e.scatter_reduce(
            dim = 0,
            index = edge_index[1],
            src = e,
            reduce='amax'
        )
        """
        找每组最大值
        max_e[0] = max(e[2], e[5])   # 节点 0 的入边来自 1→0 和 3→0
        max_e[1] = e[0]              # 节点 1 只有 0→1 一条入边
        max_e[2] = max(e[1], e[3])   # 节点 2 有 0→2 和 1→2 两条入边
        max_e[3] = e[4]              # 节点 3 只有 2→3 一条入边
        """

        # 广播减法
        # e:每条边的原始分数[E]
        # max_e:每个节点的最大入边分数[N]
        # e_stable:每条边分数减去所属分组的最大值[E]
        """
        边极数值稳定化
        e_stable[0] = e[0] - max_e[1]
        e_stable[1] = e[1] - max_e[2]
        e_stable[2] = e[2] - max_e[0]
        e_stable[3] = e[3] - max_e[2]
        e_stable[4] = e[4] - max_e[3]
        e_stable[5] = e[5] - max_e[0]
        """
        e_stable = e - max_e[edge_index[1]]
        # 防溢出 ：如果`e` 里出现很大的正数（比如 1000），`torch.exp(1000)` 会变成`inf` ，接着`inf / inf = NaN` 。
        # 减去分组最大值后，每个分组里最大的那个变成`0` ，其余都是负数。
        # 数学上等价 ：softmax 满足`softmax(x) = softmax(x - c)` （c 为常数），
        # 所以按组各减各的最大值不改变最终归一化结果。
        
        # 计算 exp
        exp_e = torch.exp(e_stable)
        
        # 计算每个节点的分母 (邻居 exp 之和)
        # sum_exp(4,)
        # 每个节点的分母是其所有入边的 exp 之和
        sum_exp = torch.zeros(N, dtype=e.dtype, device=e.device)
        sum_exp = sum_exp.scatter_add(0, edge_index[1], exp_e)
        """
        每组指数和
        sum_exp[0] = exp_e[2] + exp_e[5]   # 节点 0 的入边：1→0, 3→0
        sum_exp[1] = exp_e[0]              # 节点 1 的入边：0→1
        sum_exp[2] = exp_e[1] + exp_e[3]   # 节点 2 的入边：0→2, 1→2
        sum_exp[3] = exp_e[4]              # 节点 3 的入边：2→3
        """
        
        # 计算最终的注意力系数 alpha_ij: [E]
        alpha = exp_e / (sum_exp[edge_index[1]] + 1e-10) # 加极小值防除零
        
        # Step 4: 聚合 (Aggregation) - 稀疏矩阵乘法
        # 将注意力系数乘到源节点特征上
        # Wh:(4,5)
        # Wh[edge_index[0]]:(6,5)
        # alpha 需要扩展维度以匹配特征维度: [E, 1]
        # 广播机制 ：[E, F'] * [E, 1] -> [E, F']
        # 这行是把注意力权重“贴”到源节点特征上 ，生成每条边的“消息”（message）
        weighted_features = Wh[edge_index[0]] * alpha.unsqueeze(1) # [E, out_features]
        # weighted_features = edge_h_src * alpha.unsqueeze(1)
        # 为什么乘的是源节点而不是目标节点 ：这是消息传递（message passing）的标准约定——“节点 i 从邻居 j 那里收到消息，消息内容取自 j 的特征，强度由`alpha_ij` 决定”。
        # `alpha` 回答的是「源 j 对目标 i 有多重要」，所以被加权的是 源 的特征。
        
        # 按目标节点求和 (Scatter Add)
        # out_features(4,5)
        out_features = torch.zeros(N, Wh.shape[1], dtype=Wh.dtype, device=Wh.device)
        out_features = out_features.scatter_add(
            dim = 0, 
            index = edge_index[1].unsqueeze(1).expand(-1, Wh.shape[1]), 
            src = weighted_features
        )

        """
        out[0] = α₁₀ · Wh[1] + α₃₀ · Wh[3]      # 节点 0 的入边来自 1 和 3
        out[1] = α₀₁ · Wh[0]                    # 节点 1 的入边来自 0
        out[2] = α₀₂ · Wh[0] + α₁₂ · Wh[1]      # 节点 2 的入边来自 0 和 1
        out[3] = α₂₃ · Wh[2]                    # 节点 3 的入边来自 2
        """
        
        return out_features


# ================= 运行 Demo =================

if __name__ == "__main__":
    # 1. 构造一个简单图
    # 假设有 4 个节点，每个节点特征维度为 3
    N = 4
    in_features = 3
    out_features = 5
    
    # 随机生成节点特征(4,3)
    h = torch.randn(N, in_features)
    
    # 定义边列表 (源节点 -> 目标节点)
    # 0->1, 0->2, 1->0, 1->2, 2->3, 3->0
    # 注意：GAT 通常建议包含自环 (Self-loop)，这里为了演示简化，暂不加
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 3],  # 源节点
        [1, 2, 0, 2, 3, 0]   # 目标节点
    ], dtype=torch.long)
    
    print(f"输入节点特征维度: {h.shape}")
    print(f"边数量: {edge_index.shape[1]}")

    
    # 2. 实例化 GAT 层
    gat_layer = SingleHeadGATLayer(in_features, out_features)
    
    with torch.no_grad():
        # 3. 前向传播
        output = gat_layer(h, edge_index)
    
    print(f"\n输出节点特征维度: {output.shape}")
    print("输出所有节点特征:")
    print(output)