#!/usr/bin/env python3
"""推理脚本."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm_train.model.llama import LlamaForCausalLM
from llm_train.tokenizer import get_tokenizer
from llm_train.inference import TextGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_sample", action="store_true")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if os.path.isdir(ckpt):
        model = LlamaForCausalLM.from_pretrained(ckpt, map_location="cpu")
    else:
        from llm_train.model.config import ModelConfig
        cfg_dir = os.path.dirname(os.path.dirname(ckpt))
        cfg_path = os.path.join(cfg_dir, "config.json")
        cfg = ModelConfig.load(cfg_path) if os.path.exists(cfg_path) else None
        if cfg is None:
            raise FileNotFoundError(f"找不到 {cfg_path}; 请传目录")
        model = LlamaForCausalLM(cfg)
        sd = torch.load(ckpt, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd)

    tk = get_tokenizer(args.tokenizer, kind="bpe")
    gen = TextGenerator(model, tk)

    if args.stream:
        for chunk in gen.stream(args.prompt, max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p):
            print(chunk, end="", flush=True)
        print()
    else:
        out = gen.generate([args.prompt], max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                           do_sample=not args.no_sample)
        print(out[0])


if __name__ == "__main__":
    main()
