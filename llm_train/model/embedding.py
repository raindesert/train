"""Token Embedding + Rotary Position Embedding (RoPE)."""
from __future__ import annotations
import math

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def rotate_half(x):
    """把最后一维 (head_dim) 分两半,后一半取负并交换位置."""
    if _HAS_TORCH:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    else:
        import numpy as np
        half = x.shape[-1] // 2
        return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary_emb(q, k, cos, sin):
    """对 q/k 应用旋转位置编码."""
    if _HAS_TORCH:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
    else:
        import numpy as np
        cos = cos[np.newaxis, np.newaxis]
        sin = sin[np.newaxis, np.newaxis]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


if _HAS_TORCH:

    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            nn.init.normal_(self.embed.weight, std=0.02)
        def forward(self, ids):
            return self.embed(ids)

    class RotaryEmbedding(nn.Module):
        def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
            super().__init__()
            if head_dim % 2 != 0:
                raise ValueError(f"head_dim 必须为偶数, 得到 {head_dim}")
            self.head_dim = head_dim
            self.max_seq_len = max_seq_len
            self.theta = theta
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(max_seq_len)

        def _set_cos_sin_cache(self, seq_len):
            self.max_seq_len = max(seq_len, self.max_seq_len)
            t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos_cached", emb.cos().to(torch.float32), persistent=False)
            self.register_buffer("sin_cached", emb.sin().to(torch.float32), persistent=False)

        def forward(self, x, seq_len, offset=0):
            """返回 cos/sin, 可带位置偏移 (KV cache 推理时 offset=num_items)."""
            cache_len = seq_len + offset
            if cache_len > self.max_seq_len:
                self._set_cos_sin_cache(cache_len)
            cos = self.cos_cached[offset:offset + seq_len].to(x.dtype)
            sin = self.sin_cached[offset:offset + seq_len].to(x.dtype)
            return cos, sin

else:
    # numpy stubs
    class TokenEmbedding:
        def __init__(self, vocab_size, hidden_size): pass
        def forward(self, ids): pass

    class RotaryEmbedding:
        def __init__(self, head_dim, max_seq_len=8192, theta=10000.0):
            import numpy as np
            if head_dim % 2 != 0:
                raise ValueError("head_dim must be even")
            self.head_dim = head_dim
            self.max_seq_len = max_seq_len
            self.theta = theta
            inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype="float32") / head_dim))
            self.inv_freq = inv_freq

        def forward(self, x, seq_len):
            import numpy as np
            t = np.arange(max(seq_len, self.max_seq_len), dtype="float32")
            freqs = np.einsum("i,j->ij", t, self.inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            return emb[:seq_len].copy(), emb[:seq_len].copy()
