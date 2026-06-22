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

    def test_kv_heads_gt_q_heads_raises(self):
        """GQA 语义约束: num_kv_heads 不能大于 num_heads (MHA 是 num_kv_heads==num_heads 特例)。

        反例会让 num_kv_groups = 0, attention 计算会广播失败。
        """
        with self.assertRaises(ValueError) as ctx:
            ModelConfig(vocab_size=50, hidden_size=32, num_layers=2,
                        num_heads=2, num_kv_heads=4)
        msg = str(ctx.exception)
        self.assertIn("num_kv_heads", msg)
        self.assertIn("不能大于", msg)

    def test_kv_heads_eq_q_heads_ok(self):
        """MHA 风格 (num_kv_heads == num_heads) 必须合法."""
        c = ModelConfig(vocab_size=50, hidden_size=32, num_layers=2,
                        num_heads=2, num_kv_heads=2)
        self.assertEqual(c.num_kv_groups, 1)

    def test_kv_heads_div_q_heads_ok(self):
        """GQA 风格 (num_heads % num_kv_heads == 0) 必须合法."""
        c = ModelConfig(vocab_size=50, hidden_size=64, num_layers=2,
                        num_heads=4, num_kv_heads=2)
        self.assertEqual(c.num_kv_groups, 2)

    def test_kv_heads_not_div_q_heads_raises(self):
        """GQA 不整除 (num_heads % num_kv_heads != 0) 必须抛错."""
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=50, hidden_size=32, num_layers=2,
                        num_heads=4, num_kv_heads=3)

    def test_validate_after_field_change(self):
        """字段修改后 _validate() 必须重跑 (dataclass __post_init__ 只在 init 跑一次).

        这是 test_e2e 暴露的洞: 改 num_heads 不调 _validate 不会抛错,
        构造模型时 attention 广播失败。需要手动 revalidate。
        """
        c = ModelConfig.tiny(vocab_size=50)  # num_heads=4, num_kv_heads=4
        # 修改 num_heads 不触发 __post_init__ — 当前不会抛
        c.num_heads = 2  # num_kv_heads 还是 4
        # 手动 revalidate — 必须抛
        with self.assertRaises(ValueError):
            c._validate()
        # 修配置后 _validate 应该过
        c.num_kv_heads = 2
        c._validate()  # 不抛
        self.assertEqual(c.num_kv_groups, 1)


if __name__ == "__main__":
    unittest.main()
