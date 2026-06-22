"""RMSNorm (LLaMA 风格).

优先使用 torch (支持 autograd); 不可用时退化为 numpy 参考实现 ——
仅用于无 torch 环境做语法 + 数学正确性测试。
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if _HAS_TORCH:

    class RMSNorm(nn.Module):
        """y = x / RMS(x) * weight, weight 形状 (hidden_size,)."""
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).to(x.dtype) * self.weight

else:
    class RMSNorm:  # type: ignore[no-redef]
        """numpy-only fallback."""
        def __init__(self, dim: int, eps: float = 1e-5):
            import numpy as np
            self.eps = eps
            self.weight = np.ones(dim, dtype="float32")
        def forward(self, x):
            import numpy as np
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
            return (x / rms) * self.weight
