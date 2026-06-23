"""基于 KV cache 的 Generator.

直接用 LlamaForCausalLM.generate, 这里再封装一个支持 batch 的高层接口。
"""
from __future__ import annotations
import torch
from typing import List, Optional

from ..model.attention import KVCache


class TextGenerator:
    def __init__(self, model, tokenizer, device: str = "auto"):
        from ..utils.device import get_device
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else get_device("auto")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def generate(self,
                 prompts: List[str],
                 max_new_tokens: int = 64,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 stop: Optional[List[str]] = None,
                 ) -> List[str]:
        out_texts = []
        eos = self.tokenizer.eos_id
        for p in prompts:
            ids = self.tokenizer.encode(p, add_bos=False)
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            y = self.model.generate(
                x, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, eos_token_id=eos,
            )
            # 去掉 prompt
            gen_ids = y[0, len(ids):].tolist()
            if eos is not None and eos in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos)]
            text = self.tokenizer.decode(gen_ids, skip_special=True)
            if stop:
                for s in stop:
                    if s in text:
                        text = text.split(s, 1)[0]
            out_texts.append(text)
        return out_texts

    def chat(self, prompt: str, **kw) -> str:
        return self.generate([prompt], **kw)[0]


class Generator:
    """流式生成器: 一次返回一个 token (用于交互)."""
    def __init__(self, model, tokenizer, device: str = "auto"):
        from ..utils.device import get_device
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else get_device("auto")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def stream(self, prompt: str, max_new_tokens: int = 128,
               temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9):
        ids = self.tokenizer.encode(prompt, add_bos=False)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        T = x.shape[1]
        kv_caches = [KVCache() for _ in range(self.model.cfg.num_layers)]
        out = self.model.forward(x, kv_caches=kv_caches)
        next_logits = out.logits[:, -1, :]
        eos = self.tokenizer.eos_id

        for _ in range(max_new_tokens):
            if temperature > 0 and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], float("-inf"))
            if 0 < top_p < 1:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                cum = sorted_logits.softmax(-1).cumsum(-1)
                mask = cum > top_p
                mask[..., 0] = False
                sorted_logits[mask] = float("-inf")
                next_logits = next_logits.scatter(-1, sorted_idx, sorted_logits)
            probs = (next_logits / max(temperature, 1e-5)).softmax(-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if nxt.item() == eos:
                break
            yield self.tokenizer.decode([nxt.item()], skip_special=True)
            out = self.model.forward(nxt, kv_caches=kv_caches)
            next_logits = out.logits[:, -1, :]
