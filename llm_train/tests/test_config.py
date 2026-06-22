"""ModelConfig 单元测试 — 不依赖 torch."""
import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_train.model.config import ModelConfig


class TestConfig(unittest.TestCase):
    def test_head_divisibility(self):
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=100, hidden_size=10, num_heads=3)

    def test_intermediate_size_default(self):
        c = ModelConfig(vocab_size=100, hidden_size=768, num_layers=2, num_heads=12)
        # 8/3 * 768 = 2048, round to 256 -> 2048
        self.assertEqual(c.intermediate_size, 2048)
        self.assertEqual(c.head_dim, 64)
        self.assertEqual(c.num_kv_groups, 1)

    def test_gqa(self):
        c = ModelConfig(vocab_size=100, hidden_size=512, num_layers=4, num_heads=8, num_kv_heads=2)
        self.assertEqual(c.num_kv_groups, 4)

    def test_gqa_invalid(self):
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=100, hidden_size=512, num_layers=4, num_heads=8, num_kv_heads=3)

    def test_save_load(self):
        c = ModelConfig.tiny(vocab_size=8000)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            c.save(path)
            c2 = ModelConfig.load(path)
            self.assertEqual(c2.hidden_size, c.hidden_size)
            self.assertEqual(c2.num_layers, c.num_layers)
        finally:
            os.unlink(path)

    def test_presets(self):
        for name in ["tiny", "small", "medium", "large"]:
            cfg = getattr(ModelConfig, name)()
            # 校验维度合理
            self.assertGreater(cfg.vocab_size, 0)
            self.assertGreater(cfg.hidden_size, 0)
            self.assertGreater(cfg.num_layers, 0)
            self.assertGreater(cfg.num_heads, 0)
            self.assertGreater(cfg.intermediate_size, 0)
            self.assertEqual(cfg.hidden_size % cfg.num_heads, 0)
            # 大模型应大于小模型
            if name in ("small", "medium", "large"):
                self.assertGreaterEqual(cfg.num_heads, 8)


if __name__ == "__main__":
    unittest.main()
