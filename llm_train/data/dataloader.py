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
    """从 .bin (uint16/uint32 mmap) 中随机采样 seq_len 长度窗口."""
    def __init__(self, bin_path: str, seq_len: int,
                 vocab_size: Optional[int] = None,
                 bos_id: Optional[int] = None,
                 eos_id: Optional[int] = None):
        from .packing import load_bin
        self.tokens = load_bin(bin_path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size or int(self.tokens.max()) + 1
        self.bos_id = bos_id
        self.eos_id = eos_id
        self._n = max(0, len(self.tokens) - seq_len - 1)

    def __len__(self):
        return self._n

    def __getitem__(self, idx: int):
        if _HAS_TORCH:
            i = idx % self._n
            chunk = self.tokens[i: i + self.seq_len + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            return x, y
        # numpy fallback
        i = idx % self._n
        chunk = np.array(self.tokens[i: i + self.seq_len + 1], dtype="int64")
        return chunk[:-1], chunk[1:]


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
