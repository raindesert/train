"""学习率调度: cosine / linear / constant / warmup-stable-decay (WSD).

warmup 步内从 0 线性升到 peak lr; 然后按策略下降到 min_lr。
"""
import math
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 0,
                    min_lr_ratio: float = 0.1, schedule: str = "cosine"):
    """schedule: 'cosine' / 'linear' / 'constant' / 'wsd' (warmup-stable-decay)."""
    if schedule == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    if schedule == "wsd":
        # WSD: warmup -> constant peak -> cosine decay 到 min
        decay_steps = max(1, total_steps // 5)
        stable_steps = max(1, total_steps - warmup_steps - decay_steps)

        def f(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < warmup_steps + stable_steps:
                return 1.0
            # cosine decay
            t = (step - warmup_steps - stable_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * t))
        return LambdaLR(optimizer, lr_lambda=f)

    def f(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)
        if schedule == "cosine":
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * t))
        if schedule == "linear":
            return 1.0 - (1.0 - min_lr_ratio) * t
        raise ValueError(schedule)
    return LambdaLR(optimizer, lr_lambda=f)


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]
