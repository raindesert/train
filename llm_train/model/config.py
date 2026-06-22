"""模型配置 (LLaMA 风格)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """LLaMA 风格 Transformer 配置.

    必填:
        vocab_size:      词表大小
        hidden_size:     隐藏维度 (d_model)
        num_layers:      Transformer 层数
        num_heads:       注意力头数
        max_seq_len:     最大序列长度
    常用默认:
        intermediate_size = int(hidden_size * 8/3) 然后对齐到 256 倍数 (LLaMA 风格)
    """
    vocab_size: int
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 2048
    intermediate_size: Optional[int] = None
    num_kv_heads: Optional[int] = None      # GQA, None = MHA
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    norm_eps: float = 1e-5
    dropout: float = 0.0
    bias: bool = False                       # LLaMA2 风格: 大部分线性层无 bias
    tie_word_embeddings: bool = True         # 共享 embedding / lm_head
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """校验字段约束 — init 时和手动字段修改后都可以调。

        失败抛 ValueError 附带上下文, 让错误立刻可见。
        """
        # 头维度必须整除
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size({self.hidden_size}) 必须能被 num_heads({self.num_heads}) 整除"
            )
        # FFN 维度: LLaMA 风格 — hidden*8/3, round 到 256 倍数
        if self.intermediate_size is None:
            mult = 8 / 3
            sz = int(self.hidden_size * mult)
            sz = 256 * ((sz + 255) // 256)
            self.intermediate_size = sz
        # GQA: kv heads 默认等于 num_heads (MHA)
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        # GQA 语义约束: kv heads 必须 <= q heads (MHA 是 num_kv_heads == num_heads 的特例)
        # 反例会让 num_kv_groups = 0, attention 计算会广播失败
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads({self.num_kv_heads}) 不能大于 num_heads({self.num_heads}); "
                f"GQA 要求 kv heads <= q heads (MHA 是 num_kv_heads == num_heads 的特例)"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads({self.num_heads}) 必须能被 num_kv_heads({self.num_kv_heads}) 整除"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def to_dict(self) -> dict:
        d = asdict(self)
        # 不存派生字段 (intermediate_size, num_kv_heads 由 post_init 推导)
        d.pop("intermediate_size", None)
        d.pop("num_kv_heads", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # 预设
    @classmethod
    def tiny(cls, vocab_size: int = 8000) -> "ModelConfig":
        """~15M 参数: CPU 调试 / 教学."""
        return cls(vocab_size=vocab_size, hidden_size=256, num_layers=6,
                   num_heads=4, max_seq_len=512)

    @classmethod
    def small(cls, vocab_size: int = 32000) -> "ModelConfig":
        """~125M."""
        return cls(vocab_size=vocab_size, hidden_size=768, num_layers=12,
                   num_heads=12, max_seq_len=2048)

    @classmethod
    def medium(cls, vocab_size: int = 32000) -> "ModelConfig":
        """~350M."""
        return cls(vocab_size=vocab_size, hidden_size=1024, num_layers=24,
                   num_heads=16, max_seq_len=4096,
                   intermediate_size=2752)

    @classmethod
    def large(cls, vocab_size: int = 32000) -> "ModelConfig":
        """~1.3B."""
        return cls(vocab_size=vocab_size, hidden_size=2048, num_layers=24,
                   num_heads=32, max_seq_len=4096,
                   intermediate_size=5504, num_kv_heads=8)  # GQA
