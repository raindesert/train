#!/usr/bin/env python3
"""困惑度评估脚本."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm_train.model.llama import LlamaForCausalLM
from llm_train.tokenizer import get_tokenizer
from llm_train.data import TextFileSource, build_mixed_loader
from llm_train.eval import compute_perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if os.path.isdir(ckpt):
        model = LlamaForCausalLM.from_pretrained(ckpt, map_location="cpu")
    else:
        from llm_train.model.config import ModelConfig
        cfg_dir = os.path.dirname(os.path.dirname(ckpt))
        cfg = ModelConfig.load(f"{cfg_dir}/config.json")
        model = LlamaForCausalLM(cfg)
        sd = torch.load(ckpt, map_location="cpu")["model"]
        model.load_state_dict(sd)

    tk = get_tokenizer(args.tokenizer, kind="bpe")
    src = TextFileSource(args.data)
    ds, _, _ = build_mixed_loader([src], [1.0], tk, seq_len=args.seq_len,
                                  batch_size=args.batch_size)
    m = compute_perplexity(model, ds, batch_size=args.batch_size, seq_len=args.seq_len)
    print(f"loss={m['loss']:.4f}  ppl={m['ppl']:.2f}  tokens={m['tokens']}")


if __name__ == "__main__":
    main()
