#!/usr/bin/env python3
"""训练 BPE 分词器.

用法:
    python scripts/train_tokenizer.py \
        --input data/raw/tinyshakespeare.txt \
        --output checkpoints/tokenizer \
        --vocab_size 8000 \
        --limit 1000000
"""
import argparse, os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_train.tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="训练文本文件列表")
    ap.add_argument("--output", required=True)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="每个文件最多读取行数, 避免 OOM")
    args = ap.parse_args()

    files = args.input
    if args.limit:
        # 创建临时文件, 只包含前 limit 行
        tmp_files = []
        for f in files:
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
            with open(f, encoding="utf-8") as fin:
                for i, line in enumerate(fin):
                    if i >= args.limit:
                        break
                    tmp.write(line)
            tmp.close()
            tmp_files.append(tmp.name)
        files = tmp_files
        print(f"  limit {args.limit} lines per file -> {len(tmp_files)} temp files")

    tk = BPETokenizer(vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    tk.train(files)
    tk.save(args.output)
    print(f"tokenizer saved to {args.output}")

    if args.limit:
        for tmp in tmp_files:
            os.unlink(tmp)


if __name__ == "__main__":
    main()
