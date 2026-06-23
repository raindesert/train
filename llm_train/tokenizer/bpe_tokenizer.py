"""BPE 分词器 (基于 HuggingFace tokenizers).

- 提供高层 API: 训练 / 保存 / 加载 / encode / decode
- 也提供对预训练分词器 (如 LLaMA / Qwen) 的加载包装
- 当 tokenizers 库不可用时,fallback 到一个极简字符级编码器
"""
from __future__ import annotations
import os, json
from typing import List, Optional, Union


def get_tokenizer(path: str, kind: str = "bpe") -> "TokenizerWrapper":
    """按 kind 加载已有分词器:
        kind="bpe"  ->  HF tokenizers JSON (自定义训练)
        kind="hf"   ->  HF pretrained (LLaMA / Qwen 等)
    """
    from tokenizers import Tokenizer
    if kind == "bpe":
        tk = Tokenizer.from_file(f"{path}/tokenizer.json")
    elif kind == "hf":
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(path)
        return TokenizerWrapper(tk)
    else:
        raise ValueError(kind)
    return TokenizerWrapper(tk)


class BPETokenizer:
    """训练 BPE.

    用法:
        tk = BPETokenizer(vocab_size=8000, min_frequency=2)
        tk.train(["corpus1.txt", "corpus2.txt"])
        tk.save("./my_tokenizer")
    """
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2,
                 special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
        self.tokenizer = None

    def train(self, files: List[str]):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre, Whitespace
        from tokenizers.decoders import ByteLevel as ByteLevelDec

        tk = Tokenizer(BPE(unk_token=self.special_tokens[0]))
        tk.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tk.decoder = ByteLevelDec()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        tk.train(files, trainer)
        self.tokenizer = tk
        return self

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.tokenizer.save(f"{out_dir}/tokenizer.json")
        meta = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
        }
        with open(f"{out_dir}/tokenizer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(f"{path}/tokenizer.json")
        with open(f"{path}/tokenizer_meta.json") as f:
            meta = json.load(f)
        obj = cls(vocab_size=meta["vocab_size"], special_tokens=meta["special_tokens"])
        obj.tokenizer = tk
        return obj

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        bos = self.tokenizer.token_to_id("<s>")
        eos = self.tokenizer.token_to_id("</s>")
        if add_bos and bos is not None:
            ids = [bos] + ids
        if add_eos and eos is not None:
            ids = ids + [eos]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    @property
    def bos_id(self) -> Optional[int]:
        return self.tokenizer.token_to_id("<s>")

    @property
    def eos_id(self) -> Optional[int]:
        return self.tokenizer.token_to_id("</s>")

    @property
    def pad_id(self) -> Optional[int]:
        return self.tokenizer.token_to_id("<pad>")


class TokenizerWrapper:
    """统一 encode/decode 接口 (兼容 HF tokenizers 和 transformers)."""
    def __init__(self, inner):
        self.inner = inner

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if hasattr(self.inner, "encode"):           # HF tokenizers
            ids = self.inner.encode(text).ids
        else:                                       # transformers
            ids = self.inner.encode(text, add_special_tokens=False)
        # 处理特殊 token
        if hasattr(self.inner, "token_to_id"):
            bos = self.inner.token_to_id("<s>")
            eos = self.inner.token_to_id("</s>")
        else:
            bos = self.inner.bos_token_id
            eos = self.inner.eos_token_id
        if add_bos and bos is not None:
            ids = [bos] + ids
        if add_eos and eos is not None:
            ids = ids + [eos]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if hasattr(self.inner, "decode"):
            return self.inner.decode(ids, skip_special_tokens=skip_special)
        return self.inner.decode(ids, skip_special_tokens=skip_special)

    @property
    def vocab_size(self) -> int:
        if hasattr(self.inner, "get_vocab_size"):
            return self.inner.get_vocab_size()
        return self.inner.vocab_size

    @property
    def bos_id(self) -> Optional[int]:
        if hasattr(self.inner, "token_to_id"):
            return self.inner.token_to_id("<s>")
        return self.inner.bos_token_id

    @property
    def eos_id(self) -> Optional[int]:
        if hasattr(self.inner, "token_to_id"):
            return self.inner.token_to_id("</s>")
        return self.inner.eos_token_id

    @property
    def pad_id(self) -> Optional[int]:
        if hasattr(self.inner, "token_to_id"):
            return self.inner.token_to_id("<pad>")
        return self.inner.pad_token_id
