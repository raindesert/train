"""LoRA: 低秩适配.

核心思想:  W' = W + (alpha/r) * B @ A
  * W: 冻结 (pretrained)
  * A: (r, in), kaiming_uniform_ init
  * B: (out, r), zero init
  * 因此初始 ΔW = 0, 不破坏原模型

只对 attention 的 q_proj / v_proj (或 k_proj / o_proj) 注入 ——
保留其余层全精度。
"""
from __future__ import annotations
import os
import json
import torch
import torch.nn as nn
from typing import Iterable, List


class LoRALinear(nn.Module):
    """包一个 nn.Linear, 加低秩旁路."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        # 冻结 base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def _iter_target_linears(model: nn.Module, target_modules: Iterable[str]) -> List[tuple]:
    """返回 (name, module) 列表 — 匹配 target_modules 的 nn.Linear."""
    found = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            short = name.split(".")[-1]
            if short in target_modules:
                found.append((name, m))
    return found


def apply_lora(model: nn.Module,
               target_modules: List[str] = None,
               r: int = 8,
               alpha: float = 16.0,
               dropout: float = 0.0) -> int:
    """对 model 中匹配的 nn.Linear 套 LoRA. 返回被注入模块数."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    count = 0
    for name, lin in _iter_target_linears(model, target_modules):
        parent_name = ".".join(name.split(".")[:-1])
        attr = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        lora = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora)
        count += 1
    return count


def freeze_non_lora(model: nn.Module) -> None:
    """把所有非 lora_A / lora_B 参数 requires_grad=False."""
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False


def save_lora(model: nn.Module, path: str) -> None:
    """只保存 LoRA 旁路参数 (state_dict)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lora_sd = {n: p.cpu() for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n}
    torch.save(lora_sd, path)


def load_lora(model: nn.Module, path: str, map_location: str = "cpu") -> None:
    sd = torch.load(path, map_location=map_location)
    own = dict(model.named_parameters())
    miss = [k for k in sd if k not in own]
    if miss:
        raise KeyError(f"LoRA load: 不匹配的 key {miss[:3]}…")
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].dtype))


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """把 LoRA 合并回 base Linear (推理时可用, 减少一次 matmul).
    注意: 此操作不可逆, 若还想继续训练请先 save_lora 再 merge。
    """
    for name, m in list(model.named_modules()):
        if isinstance(m, LoRALinear):
            parent_name = ".".join(name.split(".")[:-1])
            attr = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            merged = nn.Linear(m.in_features, m.out_features, bias=m.base.bias is not None)
            # 合并权重
            delta = (m.lora_B.weight @ m.lora_A.weight) * m.scaling
            merged.weight.copy_(m.base.weight + delta)
            if m.base.bias is not None:
                merged.bias.copy_(m.base.bias)
            setattr(parent, attr, merged)
    return model
