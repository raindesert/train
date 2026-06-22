"""混合精度封装."""
import torch
from contextlib import contextmanager


@contextmanager
def autocast_context(enabled: bool, device_type: str = "cuda", dtype=None):
    """简单包装 torch.amp.autocast.

    enabled=False 时退化为 no-op 上下文。
    """
    if not enabled:
        yield None
        return
    if dtype is None:
        dtype = torch.bfloat16 if device_type == "cuda" else torch.float16
    with torch.amp.autocast(device_type=device_type, dtype=dtype):
        yield dtype


def make_grad_scaler(enabled: bool, device_type: str = "cuda"):
    """GradScaler — 仅 fp16 需要, bf16 不需要 (无动态范围损失)."""
    if not enabled or device_type != "cuda":
        return None
    return torch.amp.GradScaler("cuda")
