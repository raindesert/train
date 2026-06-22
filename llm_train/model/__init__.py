"""Transformer / LLaMA 模型定义.

子模块独立可用 (numpy 友好的部分如 RMSNorm/RoPE 数学已单测)。
torch 相关的子模块 (llama / attention) 在缺失 torch 时延迟导入。
"""
from .config import ModelConfig
from .norm import RMSNorm
from .activation import silu, SwiGLU
from .embedding import TokenEmbedding, RotaryEmbedding, apply_rotary_emb, rotate_half

# torch-dependent: 延迟导入
def __getattr__(name):
    if name in ("LlamaForCausalLM", "LlamaBlock", "ModelOutput"):
        from .llama import LlamaForCausalLM, LlamaBlock, ModelOutput
        return {"LlamaForCausalLM": LlamaForCausalLM,
                "LlamaBlock": LlamaBlock,
                "ModelOutput": ModelOutput}[name]
    if name == "Attention":
        from .attention import Attention
        return Attention
    if name == "KVCache":
        from .attention import KVCache
        return KVCache
    if name == "MLP":
        from .mlp import MLP
        return MLP
    raise AttributeError(name)


__all__ = [
    "ModelConfig", "LlamaForCausalLM", "LlamaBlock", "ModelOutput",
    "Attention", "MLP", "KVCache", "RMSNorm",
    "silu", "SwiGLU",
    "TokenEmbedding", "RotaryEmbedding", "apply_rotary_emb", "rotate_half",
]
