#!/usr/bin/env python3
"""分片 tokenize 大文件并合并为单个 .bin.

用法:
    python scripts/split_tokenize.py \
        --input data/raw/large.jsonl \
        --output data/processed/large \
        --lines_per_chunk 100000 \
        --vocab_size 20000 \
        --train_tokenizer
"""
import argparse, os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from llm_train.tokenizer import BPETokenizer
from llm_train.data import JsonlSource, ChatMLSource, pack_bin, merge_bins


def detect_source(path):
    """自动检测 jsonl 格式, 返回对应 Source."""
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if first:
            obj = json.loads(first)
            if "conversations" in obj:
                return ChatMLSource(path)
    return JsonlSource(path, field="text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 jsonl 文件")
    ap.add_argument("--output", required=True, help="输出 .bin 路径（不含扩展名）")
    ap.add_argument("--lines_per_chunk", type=int, default=100000)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--train_tokenizer", action="store_true", help="是否训练 tokenizer")
    ap.add_argument("--tokenizer_output", default="checkpoints/tokenizer")
    ap.add_argument("--tokenizer_limit", type=int, default=500000, help="训练 tokenizer 的最多行数")
    args = ap.parse_args()

    out_bin = args.output + ".bin"
    out_dir = os.path.dirname(out_bin) or "."

    # 1. 训练或加载 tokenizer
    if args.train_tokenizer or not os.path.exists(args.tokenizer_output):
        print(f"  training tokenizer (vocab_size={args.vocab_size}, limit={args.tokenizer_limit} lines)...")
        tk = BPETokenizer(vocab_size=args.vocab_size, min_frequency=args.min_frequency)
        # 用前 tokenizer_limit 行训练
        limit_file = out_dir + "/_tokenizer_train_tmp.jsonl"
        with open(args.input, encoding="utf-8") as fin, open(limit_file, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= args.tokenizer_limit:
                    break
                fout.write(line)
        tk.train([limit_file])
        tk.save(args.tokenizer_output)
        os.unlink(limit_file)
        print(f"  tokenizer saved to {args.tokenizer_output}")
    else:
        print(f"  loading tokenizer from {args.tokenizer_output}...")
        tk = BPETokenizer.load(args.tokenizer_output)

    # 2. 分片 tokenize
    chunk_bins = []
    tmp_dir = out_dir + "/_chunks"
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"  splitting & tokenizing (lines_per_chunk={args.lines_per_chunk})...")
    with open(args.input, encoding="utf-8") as f:
        chunk_idx = 0
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if len(lines) >= args.lines_per_chunk:
                chunk_file = os.path.join(tmp_dir, f"_chunk{chunk_idx}.jsonl")
                with open(chunk_file, "w", encoding="utf-8") as cf:
                    cf.writelines(lines)
                chunk_bins.append(chunk_file)
                lines = []
                chunk_idx += 1
                if chunk_idx % 20 == 0:
                    print(f"    {chunk_idx} chunks written...")
        if lines:
            chunk_file = os.path.join(tmp_dir, f"_chunk{chunk_idx}.jsonl")
            with open(chunk_file, "w", encoding="utf-8") as cf:
                cf.writelines(lines)
            chunk_idx += 1
            chunk_bins.append(chunk_file)

    print(f"  total {chunk_idx} chunks, tokenizing...")

    # 3. 每个 chunk 单独 tokenize 到临时 bin
    temp_bins = []
    for i, chunk_file in enumerate(chunk_bins):
        src = detect_source(chunk_file)
        chunk_bin = os.path.join(tmp_dir, f"_tokens{i}.bin")

        def token_gen(src=src):
            for doc in src:
                yield from tk.encode(doc)

        pack_bin(chunk_bin, token_gen(), vocab_size=tk.vocab_size)
        temp_bins.append(chunk_bin)
        os.unlink(chunk_file)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(chunk_bins)} chunks tokenized")

    # 4. 合并所有 chunk bin
    print(f"  merging {len(temp_bins)} bins...")
    merge_bins(temp_bins, out_bin, vocab_size=tk.vocab_size)

    # cleanup
    if os.path.exists(tmp_dir):
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    print(f"  done -> {out_bin}")


if __name__ == "__main__":
    main()