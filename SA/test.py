import torch

def build_causal_mask(T_q, T_k):
    """
    构造因果掩码（下三角）
    T_q: query 序列长度
    T_k: key 序列长度
    返回: (T_q, T_k) 的 bool 张量
    """
    causal = torch.ones(T_q, T_k, dtype=torch.bool).tril(diagonal=T_k - T_q)
    return causal


def show(T_q, T_k):
    mask = build_causal_mask(T_q, T_k)
    print(f"\n{'='*50}")
    print(f"T_q = {T_q},  T_k = {T_k}")
    print(f"diagonal = T_k - T_q = {T_k - T_q}")
    print(f"mask.shape = {tuple(mask.shape)}")
    print("mask (True=可见, False=屏蔽):")
    # 用 1/0 打印更直观
    print(mask.int())
    # 验证：每一行可见的 key 数量
    print("每行可见 key 的数量:", mask.sum(dim=-1).tolist())


if __name__ == "__main__":
    # 情况 1：T_q == T_k（标准自注意力 / 因果掩码）
    show(4, 4)

    # 情况 2：T_q < T_k（例如 decode 阶段，query 更短）
    show(2, 5)

    # 情况 3：T_q > T_k（例如 query 更长）
    show(5, 3)

    # 情况 4：极端，T_q = 1（自回归生成单步）
    show(1, 6)

    # 情况 5：T_k = 1
    show(4, 1)