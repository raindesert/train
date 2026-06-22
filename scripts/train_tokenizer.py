#!/usr/bin/env python3
"""训练 BPE 分词器.

用法:
    python scripts/train_tokenizer.py \
        --input data/raw/tinyshakespeare.txt \
        --output checkpoints/tokenizer \
        --vocab_size 8000
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_train.tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="训练文本文件列表")
    ap.add_argument("--output", required=True)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--min_frequency", type=int, default=2)
    args = ap.parse_args()

    tk = BPETokenizer(vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    tk.train(args.input)
    tk.save(args.output)
    print(f"tokenizer saved to {args.output}")


if __name__ == "__main__":
    main()
