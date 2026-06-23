"""多数据源混合 (text file / jsonl / huggingface)."""
from __future__ import annotations
import os
import json
import random
from typing import Iterator, List, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class _BaseSource:
    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError


class TextFileSource(_BaseSource):
    """逐行读取文本, 自动拼接达到近似 max_chars 后 yield."""
    def __init__(self, path: str, max_chars: int = 100_000):
        self.path = path
        self.max_chars = max_chars

    def __iter__(self):
        buf = []
        total = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                buf.append(line)
                total += len(line)
                if total >= self.max_chars:
                    yield "\n".join(buf)
                    buf, total = [], 0
        if buf:
            yield "\n".join(buf)


class JsonlSource(_BaseSource):
    """jsonl, 字段名可指定."""
    def __init__(self, path: str, field: str = "text"):
        self.path = path
        self.field = field

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if self.field in obj:
                    yield obj[self.field]


class ChatMLSource(_BaseSource):
    """ChatML 格式 jsonl: {"conversations": [{"role": "...", "content": "..."}, ...]}

    将 conversations 数组拼装为单一字符串, 格式:
        <|im_start|>system\n<content><|im_end|>
        <|im_start|>user\n<content><|im_end|>
        <|im_start|>assistant\n<content><|im_end|>
    """
    def __init__(self, path: str, conversations_field: str = "conversations",
                 system_field: Optional[str] = None):
        self.path = path
        self.conversations_field = conversations_field
        self.system_field = system_field

    def _format_role(self, role: str) -> str:
        return f"<|im_start|>{role}"

    def _format_msg(self, role: str, content: str) -> str:
        return f"{self._format_role(role)}\n{content}<|im_end|>"

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                convs = obj.get(self.conversations_field, [])
                parts = []
                for msg in convs:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if not content:
                        continue
                    parts.append(self._format_msg(role, content))
                if parts:
                    yield "\n".join(parts)


class HFDatasetSource(_BaseSource):
    """包装一个 HF dataset, 在 __iter__ 中拉取."""
    def __init__(self, name: str, split: str = "train", field: str = "text",
                 cache_dir: Optional[str] = None, streaming: bool = True,
                 config: Optional[str] = None):
        self.name = name
        self.split = split
        self.field = field
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.config = config

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(self.name, self.config, split=self.split,
                          cache_dir=self.cache_dir, streaming=self.streaming)
        for ex in ds:
            if self.field in ex:
                yield ex[self.field]


def build_mixed_loader(sources, weights, tokenizer,
                       seq_len: int, batch_size: int,
                       epochs: Optional[int] = None,
                       bos_token: Optional[str] = "<s>",
                       eos_token: Optional[str] = "</s>"):
    """把多个 source 按权重混合, 用 tokenizer encode 成 token 流,
    拼接到一个 .bin, 返回 TokenDataset + DataLoader.

    返回: (dataset, dataloader, bin_path)
    """
    from .packing import pack_bin
    from .dataloader import TokenDataset, DataLoader

    bos_id = tokenizer.bos_id if bos_token else None
    eos_id = tokenizer.eos_id if eos_token else None

    rng = random.Random(0)
    weights = list(weights)
    norm = sum(weights)
    weights = [w / norm for w in weights]

    bins = []
    for src, w in zip(sources, weights):
        docs = list(src)
        if w < 1.0 and len(docs) > 1:
            n = max(1, int(len(docs) * w))
            rng.shuffle(docs)
            docs = docs[:n]
        ids = []
        for d in docs:
            enc = tokenizer.encode(d, add_bos=(bos_id is not None), add_eos=(eos_id is not None))
            ids.extend(enc)
        bins.append(ids)
        print("  source %s: %d docs, %d tokens" % (src.__class__.__name__, len(docs), len(ids)))

    all_tokens = []
    rng2 = random.Random(1)
    while bins:
        idx = rng2.randrange(len(bins))
        if not bins[idx]:
            bins.pop(idx)
            continue
        chunk = min(2048, len(bins[idx]))
        all_tokens.extend(bins[idx][:chunk])
        del bins[idx][:chunk]

    out_dir = "./data/processed"
    os.makedirs(out_dir, exist_ok=True)
    # 用 token 总数命名, 避免重复迭代已耗尽的 source
    bin_path = "%s/mixed_%d.bin" % (out_dir, len(all_tokens))
    pack_bin(bin_path, all_tokens, vocab_size=tokenizer.vocab_size)

    ds = TokenDataset(bin_path, seq_len=seq_len, vocab_size=tokenizer.vocab_size,
                      bos_id=bos_id, eos_id=eos_id)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return ds, dl, bin_path