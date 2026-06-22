"""GQA (Grouped-Query) Attention + KV Cache.

实现要点:
  * 支持 MHA (num_kv_heads == num_heads) 和 GQA (num_kv_heads < num_heads)
  * 支持训练模式 (is_causal=True) 和推理模式 (KV cache 增量写入)
  * PyTorch 2.0+ 用 SDPA (scaled_dot_product_attention) 做 flash attention 加速
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .embedding import RotaryEmbedding, apply_rotary_emb


class KVCache:
    """简单的 KV cache 容器: k/v 各 (B, H_kv, T, D)."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def num_items(self) -> int:
        if self.k is None:
            return 0
        return self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """追加新 k/v,返回拼接后的完整历史."""
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None


class Attention(nn.Module):
    """LLaMA 风格多头注意力 + RoPE + GQA + KV Cache.

    投影:
      q_proj: hidden -> num_heads * head_dim
      k_proj/v_proj: hidden -> num_kv_heads * head_dim
      o_proj: num_heads * head_dim -> hidden
    """
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, max_seq_len: int = 8192,
                 rope_theta: float = 10000.0, dropout: float = 0.0,
                 bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        # 不要在 attention 内部做 dropout,用 SDPA 的 dropout 参数
        self.rotary = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(self,
                x: torch.Tensor,                  # (B, T, hidden)
                attention_mask: Optional[torch.Tensor] = None,  # (B, T) — 1 看,0 mask
                kv_cache: Optional[KVCache] = None,
                is_causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # KV cache (推理时)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # GQA: 把 k/v 沿 head 维 repeat 到 num_heads 个
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # 构造 attn_mask (B, T_q, T_k) — True/1 表示可见
        # 训练时 T_q == T_k, 用 is_causal=True 让 SDPA 内部做 causal mask
        # 推理增量时 T_q=1, 历史 T_k 已知, mask 仅用于 attention_mask
        attn_mask = None
        if attention_mask is not None:
            # (B, T_k) -> (B, 1, 1, T_k)
            attn_mask = attention_mask[:, None, None, :].to(q.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min

        # PyTorch 2.0+: SDPA 自动选择 flash / mem-efficient / math 实现
        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal and attn_mask is None,
            )
        except Exception:
            # fallback: 手写实现
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T_k)
            if is_causal and attn_mask is None:
                mask = torch.ones(T, k.shape[2], device=q.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
            if attn_mask is not None:
                scores = scores + attn_mask
            probs = scores.softmax(dim=-1)
            out = probs @ v

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
