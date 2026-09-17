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
        N = h.shape[0]          # 节点数量=4
        E = edge_index.shape[1] # 边数量=6
        src_node_index = edge_index[0]    # 源节点索引:[0,0,1,1,2,3]
        tgt_node_index = edge_index[1]    # 目标节点索引:[1,2,0,2,3,0]
        
        # Step 1: 线性变换 (所有节点并行)
        # h(4,3)
        # Wh(4,5)
        Wh = self.W(h)
        """
        四个节点的特征表示
        Wh[0] = [ft_0, ft_1, ft_2, ft_3, ft_4]
        Wh[1] = [ft_0, ft_1, ft_2, ft_3, ft_4]
        Wh[2] = [ft_0, ft_1, ft_2, ft_3, ft_4]
        Wh[3] = [ft_0, ft_1, ft_2, ft_3, ft_4]
        """
        
        # Step 2: 拼接目标节点特征和源节点特征
        Wh_i = Wh[tgt_node_index]  # [6, 5]
        Wh_j = Wh[src_node_index]  # [6, 5]
        
        # Wh_i || Wh_j
        edge_h_cat = torch.cat([Wh_i, Wh_j], dim=1) # [6, 10]
        """
        edge_h_cat[0] = [ft_0, ft_1, ft_2, ft_3, ft_4, ft_0, ft_1, ft_2, ft_3, ft_4]  # (i,j):(1,0)
        edge_h_cat[1] = [ft_0, ft_1, ft_2, ft_3, ft_4, ft_0, ft_1, ft_2, ft_3, ft_4]  # (i,j):(2,0)
        edge_h_cat[2] = [ft_0, ft_1, ft_2, ft_3, ft_4, ft_0, ft_1, ft_2, ft_3, ft_4]  # (i,j):(0,1)
        edge_h_cat[3] = [ft_0, ft_1, ft_2, ft_3, ft_4, ft_0, ft_1, ft_2, ft_3, ft_4]  # (i,j):(2,1)
        edge_h_cat[4] = [ft_0, ft_1, ft_2, ft_3, ft_4, ft_0, ft_1, ft_2, ft_3, ft_4]  # (i,j):(3,2)
        edge_h_cat[5] = [ft_0, ft_1, ft_2, ft_3, ft_4, ft_0, ft_1, ft_2, ft_3, ft_4]  # (i,j):(0,3)
                         └──────── Wh_i ──────────┘    └──────── Wh_j────────────┘

        self.a = [w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]
        """
        
        # 计算注意力logits: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        e  = self.leakyrelu(torch.matmul(edge_h_cat, self.a)).squeeze(1) # [E,1]->[E]
        """
        e[0]  -> note1 给note0打的分数logit
        e[1]  -> note2 给note0打的分数logit
        e[2]  -> note0 给note1打的分数logit
        e[3]  -> note2 给note1打的分数logit
        e[4]  -> note3 给note2打的分数logit
        e[5]  -> note0 给note3打的分数logit
        """
        
        # Step 3: 分段 Softmax (Segment Softmax) - 按目标节点归一化
        # 我们需要对每个目标节点 i 的所有邻居的 e_ij 做 Softmax
        # 使用 scatter_reduce 来实现按目标节点分组求最大值和指数和
        

        # include_self=False 让初值不参与归约，无需再构造"-1e9"这类哨兵值。
        # 初值仅为"无入边节点"占位，而这类节点不会出现在 tgt_node_index 中，
        # 因此该位置永远不会被 max_e[tgt_node_index] 读取到。
        max_e = torch.zeros(N, dtype=e.dtype, device=e.device)
        max_e = max_e.scatter_reduce(
            dim = 0,
            index = tgt_node_index, # (1,2,0,2,3,0)
            src = e,
            reduce='amax',
            include_self=False # 不包括max_e初始值0
        )
        """
        max_e[0] = max(e[2], e[5])   # 节点 0 的入边来自 1→0 和 3→0
        max_e[1] = e[0]              # 节点 1 只有 0→1 一条入边
        max_e[2] = max(e[1], e[3])   # 节点 2 有 0→2 和 1→2 两条入边
        max_e[3] = e[4]              # 节点 3 只有 2→3 一条入边
        """


        # 每条边的logit - 组内最大值logit
        # 减去最大值防止溢出
        e_stable = e - max_e[tgt_node_index]
        """
        边极数值稳定化
        e_stable[0] = e[0] - max_e[1]
        e_stable[1] = e[1] - max_e[2]
        e_stable[2] = e[2] - max_e[0]
        e_stable[3] = e[3] - max_e[2]
        e_stable[4] = e[4] - max_e[3]
        e_stable[5] = e[5] - max_e[0]
        """
        # 防溢出 ：如果`e` 里出现很大的正数（比如 1000），`torch.exp(1000)` 会变成`inf` ，接着`inf / inf = NaN` 。
        # 减去分组最大值后，每个分组里最大的那个变成`0` ，其余都是负数。
        # 数学上等价 ：softmax 满足`softmax(x) = softmax(x - c)` （c 为常数），
        # 所以按组各减各的最大值不改变最终归一化结果。
        
        # 计算每条边的exp，上一步
        exp_e = torch.exp(e_stable)
        
        # 计算每个分组的exp之和
        sum_exp = torch.zeros(N, dtype=e.dtype, device=e.device)
        sum_exp = sum_exp.scatter_add(
            dim = 0, 
            index = tgt_node_index, 
            src = exp_e
        )
        """
        每个节点的分母是其所有入边的 exp 之和
        sum_exp[0] = exp_e[2] + exp_e[5]   # 节点 0 的入边：1→0, 3→0
        sum_exp[1] = exp_e[0]              # 节点 1 的入边：0→1
        sum_exp[2] = exp_e[1] + exp_e[3]   # 节点 2 的入边：0→2, 1→2
        sum_exp[3] = exp_e[4]              # 节点 3 的入边：2→3
        """
        
        # 计算每条边的softmax(组内计算)
        alpha = exp_e / (sum_exp[tgt_node_index] + 1e-10) # 加极小值防除零

        """
        alpha[0] = exp_e[0] / sum_exp[1] = exp_e[0] / (exp_e[0])
        alpha[1] = exp_e[1] / sum_exp[2] = exp_e[1] / (exp_e[1] + exp_e[3])
        alpha[2] = exp_e[2] / sum_exp[0] = exp_e[2] / (exp_e[2] + exp_e[5])
        alpha[3] = exp_e[3] / sum_exp[2] = exp_e[3] / (exp_e[1] + exp_e[3])
        alpha[4] = exp_e[4] / sum_exp[3] = exp_e[4] / (exp_e[4])
        alpha[5] = exp_e[5] / sum_exp[0] = exp_e[5] / (exp_e[2] + exp_e[5])
        """
        
        # Step 4: 权重*源节点特征
        # Wh_j:(6,5)
        # alpha(6,)->(6,1)
        # weighted_features(6,5)
        weighted_features = Wh_j * alpha.unsqueeze(1) # 逐元素相乘，不是矩阵乘法

        
        # 按目标节点求和 (Scatter Add)
        # out_features(4,5)
        out_features = torch.zeros((N, Wh.shape[1]), dtype=Wh.dtype, device=Wh.device)
        out_features = out_features.scatter_add(
            dim = 0, 
            index = tgt_node_index.unsqueeze(1).expand(-1, Wh.shape[1]), # (6,5)
            src = weighted_features
        )
        # index、src这两个张量的shape必须相同，src通过下标去index里找地址
        # out的dim维度独立(属于归属维度)，其他维度和src保持一致
        # index是路由
        # src里的每个元素，经过index，路由到out对应的位置
        # 路由发生在dim维度，其他维度out和src保持一致

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
    
    print("输入节点特征:")
    print(h)
    print(f"输入节点特征维度: {h.shape}")
    print(f"边数量: {edge_index.shape[1]}")

    
    # 2. 实例化 GAT 层
    single_gat_model = SingleHeadGATLayer(in_features, out_features)
    
    with torch.no_grad():
        # 3. 前向传播
        # 3. 前向传播
        # h: (4, 3)
        # edge_index: (2, 6)
        # output: (4, 5)
        output = single_gat_model(h, edge_index)
    
    print(f"\n输出节点特征维度: {output.shape}")
    print("输出所有节点特征:")
    print(output)