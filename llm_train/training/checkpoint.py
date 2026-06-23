"""Checkpoint 保存与恢复."""
import os
import torch
from typing import Optional, Dict


def save_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None,
                    scaler: Optional[torch.amp.GradScaler] = None,
                    step: int = 0,
                    epoch: int = 0,
                    metrics: Optional[Dict] = None,
                    extra: Optional[Dict] = None) -> None:
    """原子保存 (临时文件 + rename)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra:
        state["extra"] = extra
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None,
                    scaler=None,
                    map_location: str = "cpu") -> Dict:
    # weights_only=False: checkpoint 含 optimizer/scheduler state (非纯 tensor)
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state
