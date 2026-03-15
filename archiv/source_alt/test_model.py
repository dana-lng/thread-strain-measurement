import torch
from model import build_model

for ch in [16, 32]:
    m = build_model("standard", base_ch=ch)
    params = sum(p.numel() for p in m.parameters())
    x = torch.zeros(2, 1, 1000, 128)
    y = m(x)
    print(f"  base_ch={ch}: {params:,} Parameter  Output={tuple(y.shape)}")

print("OK")
