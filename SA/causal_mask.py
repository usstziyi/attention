import torch

def build_causal_mask(T_q, T_k):
    one = torch.ones(T_q, T_k, dtype=torch.int32)
    # Tensor.tril(diagonal=d) 的语义是： 保留列索引 j 满足`j <= i + d` 的元素，其余置 0 （i 是行索引，d=0 就是标准下三角）。
    # 目的：右下角对齐，保留 query 序列的元素，其他置 0，实现因果掩码
    print(f"T_k - T_q = {T_k - T_q}")
    causal = one.tril(diagonal=T_k - T_q)
    return one, causal


def show(T_q, T_k):
    one, mask = build_causal_mask(T_q, T_k)
    print(one)
    print(mask)



if __name__ == "__main__":
    print("="*50)
    show(4, 4)
    print("="*50)
    show(4, 3)
    print("="*50)
    show(3, 4)
    print("="*50)
