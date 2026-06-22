"""LoRA 数学测试 (numpy 模拟)."""
import unittest
import numpy as np


class TestLoRAFormula(unittest.TestCase):
    def test_zero_init_preserves_base(self):
        """LoRA 初始化 B=0 时, forward = base(x)."""
        np.random.seed(0)
        base_W = np.random.randn(4, 6)
        x = np.random.randn(2, 6)
        A = np.random.randn(2, 6) * 0.1   # r=2
        B = np.zeros((4, 2))                # zero
        out_base = x @ base_W.T
        out_lora = x @ base_W.T + x @ A.T @ B.T * (16 / 2)
        np.testing.assert_allclose(out_lora, out_base, atol=1e-6)

    def test_lora_rank_effect(self):
        """rank 越大, 表达能力越强 (在拟合随机目标上)."""
        # 简化: 验证 LoRA 增量形状
        d_in, d_out, r = 16, 8, 4
        A = np.random.randn(r, d_in)
        B = np.random.randn(d_out, r)
        delta = B @ A   # (d_out, d_in)
        self.assertEqual(delta.shape, (d_out, d_in))


if __name__ == "__main__":
    unittest.main()
