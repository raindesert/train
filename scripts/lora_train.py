#!/usr/bin/env python3
"""LoRA 微调脚本."""
import argparse, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm_train.model.config import ModelConfig
from llm_train.model.llama import LlamaForCausalLM
from llm_train.tokenizer import get_tokenizer
from llm_train.lora import apply_lora, freeze_non_lora, save_lora
from llm_train.training import Trainer, TrainerConfig
from llm_train.data import JsonlSource, TextFileSource, build_mixed_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--field", default="text")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tk = get_tokenizer(args.tokenizer, kind="bpe")
    cfg["model"]["vocab_size"] = tk.vocab_size

    model_cfg = ModelConfig.from_dict(cfg["model"])
    model = LlamaForCausalLM(model_cfg)

    # 加载预训练权重
    resume = args.resume or cfg["training"].get("resume")
    if resume:
        from llm_train.training import load_checkpoint
        sd = torch.load(resume, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=False)
        print(f"loaded base from {resume}")

    # 注入 LoRA
    lcfg = cfg["lora"]
    n = apply_lora(model, target_modules=lcfg["target_modules"],
                   r=lcfg["r"], alpha=lcfg["alpha"], dropout=lcfg["dropout"])
    freeze_non_lora(model)
    print(f"LoRA injected: {n} modules, trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # data
    src = JsonlSource(args.data, field=args.field) if args.data.endswith(".jsonl") else TextFileSource(args.data)
    _, loader, _ = build_mixed_loader([src], [1.0], tk,
                                      seq_len=cfg["data"]["seq_len"],
                                      batch_size=cfg["data"]["batch_size"])

    tcfg = TrainerConfig.from_dict(cfg["training"])
    tcfg.grad_accum = cfg["data"].get("grad_accum", 1)
    trainer = Trainer(model, loader, None, tcfg)
    trainer.train()

    # 保存 LoRA
    save_lora(model, f"{tcfg.out_dir}/lora.pt")
    print(f"LoRA saved to {tcfg.out_dir}/lora.pt")


if __name__ == "__main__":
    main()
