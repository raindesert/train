#!/usr/bin/env python3
"""全参微调 (SFT) — 与 pretrain.py 共用 Trainer."""
import argparse, json, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm_train.model.config import ModelConfig
from llm_train.model.llama import LlamaForCausalLM
from llm_train.tokenizer import get_tokenizer
from llm_train.training import Trainer, TrainerConfig
from llm_train.data import JsonlSource, TextFileSource, ChatMLSource, build_mixed_loader


def make_source(path: str, field: str = "text"):
    if path.endswith(".jsonl"):
        # 自动检测 ChatML 格式 (conversations 数组)
        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                obj = json.loads(first_line)
                if "conversations" in obj:
                    return ChatMLSource(path)
        return JsonlSource(path, field=field)
    return TextFileSource(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="jsonl 或 txt 文件路径")
    ap.add_argument("--field", default="text")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tk = get_tokenizer(args.tokenizer, kind="bpe")
    cfg["model"]["vocab_size"] = tk.vocab_size

    src = make_source(args.data, field=args.field)
    _, loader, _ = build_mixed_loader([src], [1.0], tk,
                                      seq_len=cfg["data"]["seq_len"],
                                      batch_size=cfg["data"]["batch_size"])

    model_cfg = ModelConfig.from_dict(cfg["model"])
    model = LlamaForCausalLM(model_cfg)
    tcfg = TrainerConfig.from_dict(cfg["training"])
    tcfg.grad_accum = cfg["data"].get("grad_accum", 1)
    if args.resume:
        tcfg.resume = args.resume
    elif cfg["training"].get("resume"):
        tcfg.resume = cfg["training"]["resume"]
    trainer = Trainer(model, loader, None, tcfg)
    trainer.train()


if __name__ == "__main__":
    main()
