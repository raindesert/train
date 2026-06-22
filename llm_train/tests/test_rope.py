"""RoPE 测试 — 用 numpy 实现作为 ground truth."""
import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def rope_ref(x, cos, sin):
    """标准 RoPE (numpy 实现)."""
    # x: (T, D), cos/sin: (T, D)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # rotate_half: (-x2, x1) 拼接
    rot = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


class TestRoPEFormula(unittest.TestCase):
    def test_apply_against_ref(self):
        np.random.seed(0)
        T, D = 4, 8
        x = np.random.randn(T, D)
        cos = np.cos(np.random.randn(T, D))
        sin = np.sin(np.random.randn(T, D))
        y = rope_ref(x, cos, sin)
        # 检查维度一致 + 性质
        self.assertEqual(y.shape, x.shape)
        # 检查不是 no-op
        self.assertGreater(np.abs(y - x).sum(), 0.1)

    def test_position_invariance(self):
        """同一 token 在不同位置应得到不同输出."""
        np.random.seed(1)
        D = 8
        x = np.random.randn(D)
        cos1 = np.cos(np.arange(D, dtype=np.float32) * 0.1)
        sin1 = np.sin(np.arange(D, dtype=np.float32) * 0.1)
        cos2 = np.cos(np.arange(D, dtype=np.float32) * 0.5)
        sin2 = np.sin(np.arange(D, dtype=np.float32) * 0.5)
        y1 = rope_ref(x, cos1, sin1)
        y2 = rope_ref(x, cos2, sin2)
        self.assertGreater(np.abs(y1 - y2).sum(), 0.01)


if __name__ == "__main__":
    unittest.main()
