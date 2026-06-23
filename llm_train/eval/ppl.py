"""Perplexity 评估 (支持多种数据源)."""
from __future__ import annotations
import math
import torch
from typing import Optional

from ..training.amp import autocast_context


@torch.no_grad()
def compute_perplexity(model, dataset, device: str = "auto", batch_size: int = 4,
                       seq_len: int = 1024, amp: bool = False,
                       max_batches: Optional[int] = None) -> dict:
    """在 dataset 上计算平均 loss 和 perplexity."""
    from ..utils.device import get_device, get_dtype
    device = get_device(device)
    model.to(device).eval()
    dtype = get_dtype("auto") if amp else torch.float32

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        # TokenDataset 返回 (x, y, mask) 三元组
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device); y = y.to(device)
        with autocast_context(amp, device_type=device, dtype=dtype):
            out = model(input_ids=x, labels=y)
        n = (y != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tokens += n
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "tokens": 0}
    avg_loss = total_loss / total_tokens
    return {"loss": avg_loss, "ppl": math.exp(avg_loss), "tokens": total_tokens}
