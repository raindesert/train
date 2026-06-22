"""把分词后的 token 序列打包成二进制 .bin 文件 (训练时直接 mmap).

格式:
    header: 4B magic "LMTR" + 4B uint32 version + 4B uint32 vocab_size + 8B uint64 token_count = 20B
    然后是连续 uint16/uint32 token 流 (C-order)

读端零拷贝 mmap, 训练时直接索引。
"""
from __future__ import annotations
import os
import struct
from typing import Iterable

import numpy as np


MAGIC = b"LMTR"
VERSION = 1
HEADER_SIZE = 24  # 4(magic) + 12(ver,vocab,pad) + 8(token_count)


def _token_dtype(vocab_size: int):
    if vocab_size <= 0xFFFF:
        return np.uint16
    return np.uint32


def pack_bin(out_path: str, tokens: Iterable[int], vocab_size: int):
    """把 1D token 流写进 .bin, 可直接 mmap."""
    dtype = _token_dtype(vocab_size)
    arr = np.fromiter(tokens, dtype=dtype)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, vocab_size, 0))         # padding -> 12B
        f.write(struct.pack("<Q", len(arr)))                          # token_count -> 20B
        f.write(arr.tobytes(order="C"))
    print("packed %d tokens (%.1fMB) -> %s" % (len(arr), arr.nbytes/1e6, out_path))
    return out_path


def load_bin(path: str) -> np.ndarray:
    """mmap 加载 .bin, 返回 1D numpy array."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("bad magic %r in %s" % (magic, path))
        ver, vocab, _pad = struct.unpack("<III", f.read(12))
        if ver != VERSION:
            raise ValueError("unknown version %d" % ver)
        token_count = struct.unpack("<Q", f.read(8))[0]
        dtype = _token_dtype(vocab)
        arr = np.memmap(path, mode="r", dtype=dtype, offset=HEADER_SIZE,
                        shape=(token_count,))
        if vocab <= 0xFFFF:
            assert arr.dtype == np.uint16
        return arr