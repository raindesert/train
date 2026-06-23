"""模型前向/反向测试 — 需要 torch.

如果 torch 不可用, 整个 TestCase skip。
"""
import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_train.tests.conftest import HAS_TORCH

if not HAS_TORCH:
    raise unittest.SkipTest("torch not installed")

import torch
from llm_train.model.config import ModelConfig
from llm_train.model.llama import LlamaForCausalLM
from llm_train.model.embedding import apply_rotary_emb, rotate_half


class TestModel(unittest.TestCase):
    def test_forward_backward(self):
        cfg = ModelConfig.tiny(vocab_size=100)
        cfg.hidden_size = 32; cfg.num_layers = 2; cfg.num_heads = 2
        cfg.num_kv_heads = 2; cfg.intermediate_size = 64
        cfg._validate()
        m = LlamaForCausalLM(cfg)
        x = torch.randint(0, 100, (2, 8))
        y = x.clone()
        out = m(x, labels=y)
        self.assertEqual(out.logits.shape, (2, 8, 100))
        self.assertIsNotNone(out.loss)
        out.loss.backward()

    def test_param_count_positive(self):
        cfg = ModelConfig.tiny(vocab_size=100)
        m = LlamaForCausalLM(cfg)
        self.assertGreater(m.num_parameters(), 1000)

    def test_tied_embeddings(self):
        cfg = ModelConfig.tiny(vocab_size=100)
        m = LlamaForCausalLM(cfg)
        # tie 时 lm_head.weight 和 embed_tokens.embed.weight 同 id
        if cfg.tie_word_embeddings:
            self.assertIs(m.lm_head.weight, m.embed_tokens.embed.weight)

    def test_generate(self):
        cfg = ModelConfig.tiny(vocab_size=100)
        cfg.hidden_size = 32; cfg.num_layers = 2; cfg.num_heads = 2
        cfg.num_kv_heads = 2; cfg._validate()
        m = LlamaForCausalLM(cfg)
        m.eval()
        x = torch.randint(0, 100, (1, 5))
        y = m.generate(x, max_new_tokens=5, do_sample=False)
        self.assertEqual(y.shape, (1, 10))

    def test_rotate_half(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x)
        # chunk 2 -> [1,2] [3,4]; 翻转: [-3,-4,1,2]
        torch.testing.assert_close(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    def test_attention_mask(self):
        """带 padding mask 的 attention 应只 attend 到非 pad 位置."""
        from llm_train.model.llama import LlamaForCausalLM
        from llm_train.model.config import ModelConfig
        cfg = ModelConfig.tiny(vocab_size=50)
        cfg.hidden_size = 32; cfg.num_layers = 1; cfg.num_heads = 2
        cfg.num_kv_heads = 2; cfg.intermediate_size = 64
        cfg._validate()
        m = LlamaForCausalLM(cfg)
        m.eval()
        x = torch.randint(0, 50, (1, 6))
        # mask: 前 4 个可见, 后 2 个 mask
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
        with torch.no_grad():
            out_masked = m(x, attention_mask=mask).logits
            out_full = m(x).logits
        # 前 4 个位置的 logits 应相同 (因果约束 + mask 一致)
        torch.testing.assert_close(out_masked[:, :4], out_full[:, :4], atol=1e-5, rtol=1e-4)


class TestRoPEWithTorch(unittest.TestCase):
    def test_rotary_shapes(self):
        from llm_train.model.embedding import RotaryEmbedding
        rope = RotaryEmbedding(head_dim=8, max_seq_len=32)
        cos, sin = rope(torch.zeros(1, 1), seq_len=16)
        self.assertEqual(cos.shape, (16, 8))
        self.assertEqual(sin.shape, (16, 8))


class TestKVCache(unittest.TestCase):
    def test_kv_cache_update(self):
        from llm_train.model.attention import KVCache
        cache = KVCache()
        self.assertEqual(cache.num_items(), 0)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache.update(k, v)
        self.assertEqual(cache.num_items(), 3)
        cache.update(k, v)
        self.assertEqual(cache.num_items(), 6)


if __name__ == "__main__":
    unittest.main()
