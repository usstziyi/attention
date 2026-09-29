import torch
import torch.nn as nn

torch.manual_seed(42)

# 定义单层
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation='gelu',
    batch_first=True,
    norm_first=False,         # True 则使用 Pre-LN 结构
)
encoder_layer.eval()          # 关掉 dropout, 让结果可复现

# 输入: (batch=2, seq_len=5, d_model=512)
x = torch.randn(2, 5, 512)

# 右 padding:
#   句子1: 真实长度 3, 末尾 2 个位置是 <pad>
#   句子2: 真实长度 5, 没有 pad
#
# src_key_padding_mask: (batch, seq_len)
#   注意 PyTorch 的约定是 True = 屏蔽(即该位置是 <pad>), 与手写 attention 时常用的
#   "1=保留, 0=屏蔽" 相反, 写反了会得到完全相反的结果
src_key_padding_mask = torch.tensor([
    [False, False, False, True, True],     # 句子1: 后面 2 个 <pad>
    [False, False, False, False, False],   # 句子2: 无 pad
], dtype=torch.bool)

# 输出: (batch=2, seq_len=5, d_model=512)
out = encoder_layer(x, src_key_padding_mask=src_key_padding_mask)

print(f"in shape:  {x.shape}")
print(f"pad mask:  {src_key_padding_mask.shape}")
print(f"out shape: {out.shape}")


# === 验证 mask 生效: pad 位置的输入向量不应影响真实位置的输出 ===
# 只把句子1 的 <pad> 位置换成完全不同的向量, 再看真实位置(前 3 个)的输出是否变化
x2 = x.clone()
x2[0, 3:, :] = torch.randn(2, 512)

out2 = encoder_layer(x2, src_key_padding_mask=src_key_padding_mask)

print("\n[句子1] 真实位置输出是否不受 pad 输入影响:",
      torch.allclose(out[0, :3], out2[0, :3]))          # True
print("[句子1] pad 位置输出是否变化:",
      torch.allclose(out[0, 3:], out2[0, 3:]))          # False

# 句子2 没有 pad, 改句子1 的 pad 不该影响它
print("[句子2] 输出是否不受影响:",
      torch.allclose(out[1], out2[1]))                  # True

"""
补充:
1. src_key_padding_mask 屏蔽的是 key 方向, 即所有 query 都不再关注 <pad> 位置。
2. 若还想控制 "哪些 query 能被算", 用 attn_mask (src_mask), 形状 (seq_len, seq_len),
   True = 屏蔽, 例如因果掩码用 torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)。
3. 用 nn.TransformerEncoder 包多层时, mask 是在 forward 传入, 而不是构造时传入:
   encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
   out = encoder(x, src_key_padding_mask=src_key_padding_mask)
"""


# === query 方向: 把 <pad> 位置(query 行)的输出清零 ===
# src_key_padding_mask 只管 key 方向, 它不会让 <pad> 位置的输出变成 0:
# 那些位置照样走 残差 + FFN, 所以上面的第 51 行才是 False(pad 输入变了, pad 位置输出也变)。
# 想让下游(池化 / 取 last hidden state)彻底无视 <pad>, 得在输出上自己补一刀,
# 对应手写版里 query_mask 那一步。
query_mask = (~src_key_padding_mask).unsqueeze(-1)   # (B, L, 1), True = 真实 token
out_query_masked = out * query_mask                   # 广播到 (B, L, d_model)

print("\n[query 方向] 句子1 pad 位置输出是否已清零:",
      bool(out_query_masked[0, 3:].abs().max() == 0))          # True
print("[query 方向] 句子1 真实位置输出是否没被动:",
      torch.allclose(out_query_masked[0, :3], out[0, :3]))     # True
print("[query 方向] 句子2 输出是否没被动:",
      torch.allclose(out_query_masked[1], out[1]))             # True

# query_mask 还能直接拿去做 mean pooling, 天然忽略 <pad>:
pooled = out_query_masked.sum(1) / query_mask.sum(1)        # (B, d_model)
print(f"pooled shape: {pooled.shape}")
