"""通用 Trainer — 预训练 / 微调共用.

主要职责:
  * 梯度累积
  * 混合精度 (autocast + GradScaler)
  * 日志 (loss, lr, throughput)
  * 周期性 eval + checkpoint
  * 支持 accelerate / 单 GPU 简单分发
"""
from __future__ import annotations
import os
import time
import json
import math
import warnings
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple, Union, get_type_hints

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDL

from ..utils.logging import get_logger
from ..utils.device import get_device, get_dtype
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .checkpoint import save_checkpoint, load_checkpoint
from .amp import autocast_context, make_grad_scaler

log = get_logger("trainer")


@dataclass
class TrainerConfig:
    out_dir: str = "./checkpoints/run"
    max_steps: int = 1000
    eval_every: int = 200
    save_every: int = 200
    log_every: int = 20
    grad_accum: int = 1
    grad_clip: float = 1.0
    amp: bool = False
    amp_dtype: str = "auto"           # 'bf16' / 'fp16' / 'auto'
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    schedule: str = "cosine"
    seed: int = 42
    device: str = "auto"
    resume: Optional[str] = None
    max_grad_norm: float = 1.0
    eval_max_batches: Optional[int] = None  # 限制 eval 步数

    @classmethod
    def from_dict(cls, d: dict) -> "TrainerConfig":
        """从 dict 构造, 容错 + 报告问题字段。

        行为:
          * 未知字段: 不抛错, 但 ``warnings.warn`` 列出全部被忽略的字段和值
            (YAML typo 容易让训练静默用默认值, 此举让问题可见)
          * 类型校验失败: ``warnings.warn`` + 退回 dataclass 默认值
            (避免单字段写错让整个 config 起不来)
          * 容器类型 (Dict/List) 不做校验, YAML 复杂结构原样透传
          * ``None`` 对 ``Optional[T]`` 类型始终接受

        不破坏旧行为: 旧调用方不依赖被忽略字段, 仍然得到合法 TrainerConfig.
        """
        valid_fields = {f.name: f for f in dataclasses.fields(cls)}
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {f.name: f.type for f in dataclasses.fields(cls)}

        unknown_keys = {}      # k -> v, 用于一次 warn
        type_failures = []     # (k, v, err), 用于一次 warn

        clean = {}
        for k, v in d.items():
            if k not in valid_fields:
                unknown_keys[k] = v
                continue

            # 类型校验 — Optional[T] 接受 None
            target_t = hints.get(k)
            if target_t is None or v is None:
                clean[k] = v
                continue

            # 容器类型不做严格校验 (YAML 复杂结构由使用处兜底)
            origin = getattr(target_t, "__origin__", None)
            if origin in (dict, Dict, list, List, tuple, Tuple):
                clean[k] = v
                continue

            # 简单类型: int / float / bool / str
            py = _unwrap_optional(target_t)
            if py in (int, float, str, bool):
                try:
                    if py is bool:
                        # YAML 'true'/'false' -> bool; 0/1 也接受
                        if isinstance(v, bool):
                            clean[k] = v
                        elif isinstance(v, (int, float)):
                            clean[k] = bool(v)
                        elif isinstance(v, str):
                            clean[k] = v.strip().lower() in ("true", "1", "yes", "on")
                        else:
                            raise TypeError(f"bool expects scalar, got {type(v).__name__}")
                    elif py is int:
                        clean[k] = int(v) if not isinstance(v, bool) else int(v)
                    elif py is float:
                        clean[k] = float(v)
                    elif py is str:
                        clean[k] = str(v)
                except Exception as e:
                    type_failures.append((k, v, str(e)))
                continue

            # 其它类型 (e.g. 自定义 dataclass) 不校验, 原样透传
            clean[k] = v

        if unknown_keys:
            pairs = ", ".join(f"{k}={v!r}" for k, v in unknown_keys.items())
            warnings.warn(
                f"TrainerConfig.from_dict ignored {len(unknown_keys)} unknown "
                f"field(s) (likely YAML typo): {pairs}",
                stacklevel=2,
            )
        if type_failures:
            parts = ", ".join(f"{k}={v!r} ({err})" for k, v, err in type_failures)
            warnings.warn(
                f"TrainerConfig.from_dict: {len(type_failures)} field(s) had "
                f"bad type, fell back to default: {parts}",
                stacklevel=2,
            )

        return cls(**clean)


def _unwrap_optional(tp):
    """Optional[X] -> X, 其余原样返回."""
    origin = getattr(tp, "__origin__", None)
    if origin is Union:
        args = [a for a in tp.__args__ if a is not type(None)]
        if len(args) == 1:
            return _unwrap_optional(args[0])
    return tp


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 eval_loader=None,
                 cfg: TrainerConfig = None):
        self.cfg = cfg or TrainerConfig()
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # 设备 / 精度
        self.device = get_device(self.cfg.device)
        self.dtype = get_dtype(self.cfg.amp_dtype) if self.cfg.amp else torch.float32
        self.model.to(self.device)

        # 优化器 / 调度 / AMP
        self.optimizer = build_optimizer(model, lr=self.cfg.lr,
                                         weight_decay=self.cfg.weight_decay,
                                         optimizer=self.cfg.optimizer)
        # scheduler 按 optimizer step 计数, 需要除以 grad_accum
        optimizer_total = max(1, self.cfg.max_steps // max(1, self.cfg.grad_accum))
        optimizer_warmup = (self.cfg.warmup_steps // max(1, self.cfg.grad_accum)
                            if self.cfg.warmup_steps > 0 else 0)
        self.scheduler = build_scheduler(self.optimizer,
                                         total_steps=optimizer_total,
                                         warmup_steps=optimizer_warmup,
                                         min_lr_ratio=self.cfg.min_lr_ratio,
                                         schedule=self.cfg.schedule)
        self.scaler = make_grad_scaler(self.cfg.amp, self.device, self.dtype)

        # 状态
        self.step = 0
        self.epoch = 0
        self.history: List[Dict] = []
        self.best_metric = float("inf")
        self._last_batch_tokens = 0

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        if self.cfg.resume:
            log.info(f"resuming from {self.cfg.resume}")
            st = load_checkpoint(self.cfg.resume, model, self.optimizer, self.scheduler, self.scaler, map_location=self.device)
            self.step = st.get("step", 0)
            self.epoch = st.get("epoch", 0)

    def _train_step(self, batch):
        # batch 可能是 (x, y) 或 (x, y, mask)
        if len(batch) == 3:
            x, y, mask = batch
            mask = mask.to(self.device, non_blocking=True)
        else:
            x, y = batch
            mask = None
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        with autocast_context(self.cfg.amp, device_type=self.device, dtype=self.dtype):
            out = self.model(input_ids=x, labels=y, attention_mask=mask)
            loss = out.loss / self.cfg.grad_accum
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._last_batch_tokens = x.size(0) * x.size(1)
        return loss.item() * self.cfg.grad_accum

    def train(self):
        log.info(f"start training: steps={self.cfg.max_steps} device={self.device} amp={self.cfg.amp}")
        self.model.train()
        t0 = time.time()
        losses = []
        self.train_loader.set_epoch(self.epoch) if hasattr(self.train_loader, "set_epoch") else None

        while self.step < self.cfg.max_steps:
            for batch in self.train_loader:
                if self.step >= self.cfg.max_steps:
                    break

                loss = self._train_step(batch)
                losses.append(loss)

                # 梯度累积
                if (self.step + 1) % self.cfg.grad_accum == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    # grad_clip 是 YAML 使用的字段, max_grad_norm 兼容旧配置
                    clip = self.cfg.grad_clip if self.cfg.grad_clip != 1.0 else self.cfg.max_grad_norm
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                self.step += 1

                # log
                if self.step % self.cfg.log_every == 0:
                    avg = sum(losses[-self.cfg.log_every:]) / max(1, len(losses[-self.cfg.log_every:]))
                    dt = time.time() - t0
                    tps = self.step * self._last_batch_tokens / max(1, dt)
                    log.info(
                        f"step {self.step}/{self.cfg.max_steps} "
                        f"loss={avg:.4f} lr={self.scheduler.get_last_lr()[0]:.2e} "
                        f"tok/s={tps:.0f}"
                    )

                # eval
                if self.eval_loader is not None and self.step % self.cfg.eval_every == 0:
                    m = self.evaluate()
                    log.info(f"  eval @ step {self.step}: {m}")
                    self.history.append({"step": self.step, **m})
                    self.model.train()
                    if m.get("loss", float("inf")) < self.best_metric:
                        self.best_metric = m["loss"]
                        save_checkpoint(f"{self.cfg.out_dir}/best.pt",
                                        self.model, self.optimizer, self.scheduler,
                                        self.scaler, step=self.step, epoch=self.epoch, metrics=m)

                # save
                if self.step % self.cfg.save_every == 0:
                    save_checkpoint(f"{self.cfg.out_dir}/latest.pt",
                                    self.model, self.optimizer, self.scheduler,
                                    self.scaler, step=self.step, epoch=self.epoch)

            self.epoch += 1
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(self.epoch)

        # final
        save_checkpoint(f"{self.cfg.out_dir}/final.pt",
                        self.model, self.optimizer, self.scheduler,
                        self.scaler, step=self.step, epoch=self.epoch)
        with open(f"{self.cfg.out_dir}/history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        log.info(f"training done. final step={self.step} best_loss={self.best_metric:.4f}")

    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        total, count = 0.0, 0
        max_b = self.cfg.eval_max_batches or float("inf")
        for i, batch in enumerate(self.eval_loader):
            if i >= max_b:
                break
            if len(batch) == 3:
                x, y, mask = batch
                mask = mask.to(self.device)
            else:
                x, y = batch
                mask = None
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with autocast_context(self.cfg.amp, device_type=self.device, dtype=self.dtype):
                out = self.model(input_ids=x, labels=y, attention_mask=mask)
            total += out.loss.item()
            count += 1
        avg = total / max(1, count)
        return {"loss": avg, "ppl": math.exp(avg)}
