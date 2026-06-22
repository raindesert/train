"""LLaMA 风格 FFN: gate_proj / up_proj / down_proj + SwiGLU."""
import torch
import torch.nn as nn

from .activation import SwiGLU


class MLP(nn.Module):
    """SwiGLU FFN:
        y = down_proj( silu(gate_proj(x)) * up_proj(x) )
    """
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x), self.up_proj(x)))
