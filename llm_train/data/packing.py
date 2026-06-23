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


def pack_bin(out_path: str, tokens: Iterable[int], vocab_size: int, buffer_size: int = 500000):
    """把 1D token 流写进 .bin, 流式写入避免 OOM.

    写文件头时先写入 0, 写完后 seek 回去更新 token_count。
    buffer_size: 每次写入的 token 块大小, 控制内存峰值。
    """
    dtype = _token_dtype(vocab_size)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, vocab_size, 0))
        # 先占位 token_count (8 bytes)
        count_pos = f.tell()
        f.write(struct.pack("<Q", 0))

        # 流式分块写入
        buf = np.zeros(buffer_size, dtype=dtype)
        buf_idx = 0
        token_count = 0
        for tok in tokens:
            buf[buf_idx] = tok
            buf_idx += 1
            token_count += 1
            if buf_idx >= buffer_size:
                f.write(buf.tobytes(order="C"))
                buf_idx = 0
        if buf_idx > 0:
            f.write(buf[:buf_idx].tobytes(order="C"))

        # seek 回去写正确的 token_count
        f.seek(count_pos)
        f.write(struct.pack("<Q", token_count))

    size_mb = token_count * np.dtype(dtype).itemsize / 1e6
    print("packed %d tokens (%.1fMB) -> %s" % (token_count, size_mb, out_path))
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