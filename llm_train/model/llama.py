"""LLaMA 风格的 Decoder-only Transformer."""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .config import ModelConfig
from .norm import RMSNorm
from .embedding import TokenEmbedding
from .attention import Attention, KVCache
from .mlp import MLP


@dataclass
class ModelOutput:
    logits: torch.Tensor              # (B, T, vocab)
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None


class LlamaBlock(nn.Module):
    """单层: PreNorm + Attention + PreNorm + MLP."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = Attention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            rope_theta=cfg.rope_theta,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )
        self.mlp_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size, bias=cfg.bias)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                is_causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask,
                          kv_cache=kv_cache, is_causal=is_causal)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LlamaForCausalLM(nn.Module):
    """Decoder-only 语言模型."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = TokenEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # 共享 embedding / lm_head 权重 (LLaMA 7B/13B 风格)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 标准 Transformer 初始化: Linear 正态(std=0.02), Embedding 正态(std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.initializer_range)

    def forward(self,
                input_ids: torch.Tensor,                 # (B, T)
                attention_mask: Optional[torch.Tensor] = None,  # (B, T)
                labels: Optional[torch.Tensor] = None,    # (B, T) — 与 input_ids 同形状, -100 忽略
                kv_caches: Optional[List[KVCache]] = None,
                ) -> ModelOutput:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)

        is_causal = (kv_caches is None) or (kv_caches[0] is None) or kv_caches[0].num_items() == 0

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, attention_mask=attention_mask, kv_cache=cache, is_causal=is_causal)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if labels is not None:
            # next-token prediction: 把 logits 向前 shift 一位
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(logits=logits, loss=loss)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed_tokens.embed.weight.numel()
            if self.cfg.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,       # (B, T)
                 max_new_tokens: int = 64,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 do_sample: bool = True,
                 eos_token_id: Optional[int] = None,
                 ) -> torch.Tensor:
        """带 KV cache 的自回归生成 (B 维度需 == 1 时最高效)."""
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device

        # 初始化 KV cache
        kv_caches = [KVCache() for _ in range(self.cfg.num_layers)]
        # prefill
        out = self.forward(input_ids, kv_caches=kv_caches, is_causal=True)
        next_logits = out.logits[:, -1, :]  # (B, V)

        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            if do_sample and temperature > 0:
                logits = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(descending=True)
                    cum = sorted_logits.softmax(-1).cumsum(-1)
                    mask = cum > top_p
                    mask[..., 0] = False
                    sorted_logits[mask] = float("-inf")
                    logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
                probs = logits.softmax(-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_id = next_logits.argmax(-1, keepdim=True)

            generated.append(next_id)
            finished = finished | (next_id.squeeze(-1) == eos_token_id) if eos_token_id is not None else finished

            # 用上一步的 id 跑一次 decode
            out = self.forward(next_id, kv_caches=kv_caches, is_causal=True)
            next_logits = out.logits[:, -1, :]

            if finished.all():
                break

        gen = torch.cat(generated, dim=-1)
        return torch.cat([input_ids, gen], dim=-1)

    def save_pretrained(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.cfg.save(f"{path}/config.json")
        torch.save(self.state_dict(), f"{path}/model.pt")

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "LlamaForCausalLM":
        cfg = ModelConfig.load(f"{path}/config.json")
        model = cls(cfg)
        sd = torch.load(f"{path}/model.pt", map_location=map_location)
        model.load_state_dict(sd)
        return model
