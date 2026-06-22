#!/usr/bin/env python3
"""预训练入口.

流程:
    1. 读 YAML 配置 (model / data / training)
    2. 构造分词器 (从指定路径加载, 或基于 input_files 临时训练)
    3. 把原始文本打包成 .bin
    4. 构造模型 + Trainer, 训练, 评估
"""
import argparse, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm_train.model.config import ModelConfig
from llm_train.model.llama import LlamaForCausalLM
from llm_train.tokenizer import BPETokenizer, get_tokenizer
from llm_train.data import TextFileSource, build_mixed_loader
from llm_train.training import Trainer, TrainerConfig


def load_cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # 1) tokenizer
    tok_dir = cfg.get("tokenizer", {}).get("path")
    if tok_dir and os.path.exists(tok_dir):
        tk = get_tokenizer(tok_dir, kind="bpe")
    else:
        # 训练一次性 tokenizer
        inputs = cfg["tokenizer"].get("input_files") or ["data/raw/tinyshakespeare.txt"]
        bp = BPETokenizer(vocab_size=cfg["tokenizer"]["vocab_size"],
                          min_frequency=cfg["tokenizer"].get("min_frequency", 2))
        bp.train([p for p in inputs if os.path.exists(p)])
        os.makedirs("checkpoints/tokenizer", exist_ok=True)
        bp.save("checkpoints/tokenizer")
        tk = BPETokenizer.load("checkpoints/tokenizer")
        tok_dir = "checkpoints/tokenizer"

    # 同步 vocab_size 到 model
    cfg["model"]["vocab_size"] = tk.vocab_size

    # 2) data
    inputs = cfg.get("tokenizer", {}).get("input_files") or ["data/raw/tinyshakespeare.txt"]
    inputs = [p for p in inputs if os.path.exists(p)] or ["data/raw/tinyshakespeare.txt"]
    sources = [TextFileSource(p) for p in inputs]
    weights = [1.0 / len(sources)] * len(sources)
    seq_len = cfg["data"]["seq_len"]
    batch_size = cfg["data"]["batch_size"]
    _, train_loader, _ = build_mixed_loader(sources, weights, tk,
                                            seq_len=seq_len, batch_size=batch_size)

    # 简单 eval split: 用同一文件最后 10%
    eval_sources = []
    for p in inputs:
        with open(p) as f:
            data = f.read()
        cut = int(len(data) * 0.9)
        tmp = f"data/processed/eval_{os.path.basename(p)}"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w") as g:
            g.write(data[cut:])
        eval_sources.append(TextFileSource(tmp))
    _, eval_loader, _ = build_mixed_loader(eval_sources, weights, tk,
                                           seq_len=seq_len, batch_size=batch_size)

    # 3) model
    model_cfg = ModelConfig.from_dict(cfg["model"])
    model = LlamaForCausalLM(model_cfg)
    print(f"model params: {model.num_parameters()/1e6:.1f}M")

    # 4) trainer
    tcfg = TrainerConfig.from_dict(cfg["training"])
    tcfg.grad_accum = cfg["data"].get("grad_accum", 1)
    trainer = Trainer(model, train_loader, eval_loader, tcfg)
    trainer.train()


if __name__ == "__main__":
    main()
