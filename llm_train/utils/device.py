"""设备/精度工具."""
from __future__ import annotations

def get_device(prefer: str = "auto") -> str:
    """自动选择最佳设备: cuda > mps > cpu."""
    if prefer != "auto":
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_dtype(name: str = "auto"):
    """选择 dtype: auto 选 bf16 > fp16 > fp32 (按硬件支持)."""
    try:
        import torch
        if name == "auto":
            if torch.cuda.is_available():
                # Ampere+ 支持 bf16
                cap = torch.cuda.get_device_capability(0)
                if cap[0] >= 8:
                    return torch.bfloat16
                return torch.float16
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.float16
            return torch.float32
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
    except Exception:
        return None
