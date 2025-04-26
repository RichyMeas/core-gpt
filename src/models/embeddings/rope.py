import torch
from torch import nn as nn
from torch import Tensor

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        # Ensure we don't exceed the dimension size
        dim_quarter = max(1, dim // 4)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim_quarter, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim_quarter)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq) # outer product
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        # Handle case where the number of dimensions is smaller
        dim_half = x_BTHD.size(-1) // 2
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos[..., :dim_half] + x2 * sin[..., :dim_half]
        y2 = x1 * (-sin[..., :dim_half]) + x2 * cos[..., :dim_half]
        return torch.cat((y1, y2), 3).type_as(x_BTHD)