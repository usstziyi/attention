import torch
import torch.nn as nn

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

# 输入: (batch=2, seq_len=10, d_model=512)
x = torch.randn(2, 10, 512)
# torch.Size([2, 10, 512])
out = encoder_layer(x)

print(f"in shape: {x.shape}")
print(f"out shape: {out.shape}")
