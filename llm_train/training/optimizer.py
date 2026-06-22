"""优化器构建: AdamW + 权重衰减分组 (no_decay 规则)."""
import torch
import torch.nn as nn


def _separate_params(model: nn.Module, weight_decay: float):
    """把参数分成 decay / no_decay 两组:
        - 所有 Linear / Embedding 的 weight -> decay
        - bias / norm -> no_decay
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n.lower() and "embed.weight" not in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.1,
                    betas=(0.9, 0.95), eps: float = 1e-8,
                    optimizer: str = "adamw") -> torch.optim.Optimizer:
    """构造优化器. 默认 AdamW (LLaMA 风格)."""
    groups = _separate_params(model, weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    elif optimizer == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)
    elif optimizer == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    raise ValueError(optimizer)
