#!/usr/bin/env python3
"""预训练入口.

流程:
    1. 读 YAML 配置 (model / data / training)
    2. 构造分词器 (从指定路径加载)
    3. 构造数据集 (从 --bin 直接加载, 或从 input_files 打包)
    4. 构造模型 + Trainer, 训练, 评估
"""
import argparse, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm_train.model.config import ModelConfig
from llm_train.model.llama import LlamaForCausalLM
from llm_train.tokenizer import BPETokenizer, get_tokenizer
from llm_train.data import TextFileSource, JsonlSource, build_mixed_loader, TokenDataset, DataLoader
from llm_train.training import Trainer, TrainerConfig


def load_cfg(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _WrapSource:
    """Wrap a list of strings as a data source."""
    def __init__(self, docs):
        self.docs = docs
    def __iter__(self):
        yield from self.docs


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
        inputs = cfg["tokenizer"].get("input_files") or ["data/raw/tinyshakespeare.txt"]
        print("  training tokenizer on the fly (consider pre-training for reproducibility)")
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
    seq_len = cfg["data"]["seq_len"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 0)
    bin_path = cfg["data"].get("bin_path")
    eval_bin_path = cfg["data"].get("eval_bin_path")

    if bin_path and os.path.exists(bin_path):
        ds = TokenDataset(bin_path, seq_len=seq_len, vocab_size=tk.vocab_size,
                          bos_id=tk.bos_id, eos_id=tk.eos_id)
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers)
        eval_loader = None
        if eval_bin_path and os.path.exists(eval_bin_path):
            eval_ds = TokenDataset(eval_bin_path, seq_len=seq_len, vocab_size=tk.vocab_size,
                                   bos_id=tk.bos_id, eos_id=tk.eos_id)
            eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                                     drop_last=True, num_workers=num_workers)
    else:
        inputs = cfg.get("tokenizer", {}).get("input_files") or ["data/raw/tinyshakespeare.txt"]
        inputs = [p for p in inputs if os.path.exists(p)] or ["data/raw/tinyshakespeare.txt"]

        def make_source(p):
            if p.endswith(".jsonl"):
                return JsonlSource(p, field="text")
            return TextFileSource(p)

        sources = [make_source(p) for p in inputs]
        weights = [1.0 / len(sources)] * len(sources)
        _, train_loader, _ = build_mixed_loader(sources, weights, tk,
                                                seq_len=seq_len, batch_size=batch_size)

        # eval split: 取每个 source 最后 10% 文档 (不写临时文件)
        eval_docs = []
        for src in sources:
            docs = list(iter(src))
            cut = int(len(docs) * 0.9)
            eval_docs.extend(docs[cut:])
        eval_sources = [_WrapSource(eval_docs)] if eval_docs else []
        eval_loader = None
        if eval_sources:
            _, eval_loader, _ = build_mixed_loader(eval_sources, [1.0], tk,
                                                   seq_len=seq_len, batch_size=batch_size)

    # 3) model
    model_cfg = ModelConfig.from_dict(cfg["model"])
    model = LlamaForCausalLM(model_cfg)
    print(f"model params: {model.num_parameters()/1e6:.1f}M")

    # 4) trainer
    tcfg = TrainerConfig.from_dict(cfg["training"])
    if "grad_accum" in cfg.get("data", {}):
        tcfg.grad_accum = cfg["data"]["grad_accum"]
    trainer = Trainer(model, train_loader, eval_loader, tcfg)
    trainer.train()


if __name__ == "__main__":
    main()
