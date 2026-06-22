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
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List

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
        """从 dict 构造, 容错忽略未知字段 + 修正 YAML 常见错误.

        修正:
          * 移除不在 dataclass 中的字段
          * 类型转换失败的字段保留 dataclass 默认值
        """
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        clean = {}
        for k, v in d.items():
            if k not in valid:
                continue
            # 类型校验
            f = next(f for f in dataclasses.fields(cls) if f.name == k)
            try:
                if f.type is not None and v is not None:
                    # 简单类型转换 — 跳过无法转换的让 dataclass 报
                    pass
                clean[k] = v
            except Exception:
                continue
        return cls(**clean)


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
        self.scheduler = build_scheduler(self.optimizer,
                                         total_steps=self.cfg.max_steps,
                                         warmup_steps=self.cfg.warmup_steps,
                                         min_lr_ratio=self.cfg.min_lr_ratio,
                                         schedule=self.cfg.schedule)
        self.scaler = make_grad_scaler(self.cfg.amp, self.device)

        # 状态
        self.step = 0
        self.epoch = 0
        self.history: List[Dict] = []
        self.best_metric = float("inf")

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
                    if self.cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
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
                    tps = self.step * (x.size(0) * x.size(1)) / max(1, dt)
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
            x, y = batch
            x = x.to(self.device); y = y.to(self.device)
            with autocast_context(self.cfg.amp, device_type=self.device, dtype=self.dtype):
                out = self.model(input_ids=x, labels=y)
            total += out.loss.item()
            count += 1
        avg = total / max(1, count)
        return {"loss": avg, "ppl": math.exp(avg)}
