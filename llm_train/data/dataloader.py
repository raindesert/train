"""数据集 / DataLoader.

torch 不可用时提供基于 numpy 的最小实现。
"""
from __future__ import annotations
import os, math, random
import numpy as np
from typing import Optional, List

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    class Dataset:
        pass


class TokenDataset(Dataset):
    """从 .bin (uint16/uint32 mmap) 中采样 seq_len 长度窗口.

    窗口策略:
      * 每个 epoch 内, 数据集大小 = max(1, total_tokens - seq_len)
      * __getitem__(idx) 取窗口: idx % num_windows + 步进偏移
      * 偏移基于 step: step_window_size 滑动, 减少相邻窗口的高重叠

    边界检查:
      * 如果 tokens 总数 < seq_len + 1, 直接抛错 (而非 silently 返回 0 长度)
    """
    MIN_TOKENS = 16  # 至少 16 个 token 才有训练意义

    def __init__(self, bin_path: str, seq_len: int,
                 vocab_size: Optional[int] = None,
                 bos_id: Optional[int] = None,
                 eos_id: Optional[int] = None,
                 stride: Optional[int] = None):
        from .packing import load_bin
        if seq_len < 1:
            raise ValueError(f"seq_len 必须 >= 1, 得到 {seq_len}")
        self.tokens = load_bin(bin_path)
        if len(self.tokens) < self.MIN_TOKENS:
            raise ValueError(
                f"数据集 {bin_path} 仅含 {len(self.tokens)} tokens, "
                f"少于最小 {self.MIN_TOKENS}; 请提供更多数据"
            )
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"数据集 {bin_path} 含 {len(self.tokens)} tokens, "
                f"少于 seq_len+1 ({seq_len+1}); 请减小 seq_len 或增大数据"
            )
        self.seq_len = seq_len
        self.vocab_size = vocab_size or int(self.tokens.max()) + 1
        self.bos_id = bos_id
        self.eos_id = eos_id
        # 窗口数: 取 seq_len + 1 长度的窗口能取多少次
        self._n = max(1, len(self.tokens) - seq_len - 1)
        # 步长: 默认 seq_len/2 (相邻窗口 50% 重叠, 平衡采样多样性与内存)
        self.stride = stride if stride is not None else max(1, seq_len // 2)
    def __len__(self):
        return self._n

    def __getitem__(self, idx: int):
        if _HAS_TORCH:
            # stride 滑动窗口, 减少相邻 batch 的高重叠
            i = (idx * self.stride) % self._n
            chunk = self.tokens[i: i + self.seq_len + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            # attention_mask: 1 表示可见 (当前所有 token 等长, 全 1)
            mask = torch.ones(self.seq_len, dtype=torch.long)
            return x, y, mask
        # numpy fallback
        i = (idx * self.stride) % self._n
        chunk = np.array(self.tokens[i: i + self.seq_len + 1], dtype="int64")
        return chunk[:-1], chunk[1:], np.ones(self.seq_len, dtype="int64")


class DataLoader:
    """轻量 DataLoader."""
    def __init__(self, dataset, batch_size: int,
                 shuffle: bool = True, drop_last: bool = True,
                 num_workers: int = 0, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.seed = seed
        self.epoch = 0
        self._indices = list(range(len(dataset)))

    def __len__(self):
        n = len(self._indices) // self.batch_size
        return n if self.drop_last else math.ceil(n)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _shuffle_indices(self):
        rng = random.Random(self.seed + self.epoch)
        self._indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(self._indices)

    def _batch_iter(self, batch_indices):
        xs, ys = [], []
        for i in batch_indices:
            x, y = self.dataset[i]
            xs.append(x); ys.append(y)
        if _HAS_TORCH:
            return torch.stack(xs, 0), torch.stack(ys, 0)
        return np.stack(xs, 0), np.stack(ys, 0)

    def __iter__(self):
        self._shuffle_indices()
        for s in range(0, len(self._indices), self.batch_size):
            bi = self._indices[s: s + self.batch_size]
            if self.drop_last and len(bi) < self.batch_size:
                continue
            yield self._batch_iter(bi)
