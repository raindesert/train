"""激活函数: SwiGLU (LLaMA 使用的门控激活)."""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def silu(x):
    """SiLU / Swish."""
    if _HAS_TORCH:
        return F.silu(x)
    # numpy fallback
    import numpy as np
    return x / (1.0 + np.exp(-x))


if _HAS_TORCH:

    class SwiGLU(nn.Module):
        """y = silu(gate(x)) * up(x)."""
        def __init__(self):
            super().__init__()

        def forward(self, gate, up):
            return F.silu(gate) * up

else:
    class SwiGLU:  # type: ignore[no-redef]
        def forward(self, gate, up):
            return silu(gate) * up