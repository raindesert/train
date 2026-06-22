"""RMSNorm 数学正确性测试 (用 numpy)."""
import unittest
import numpy as np


def rms_norm(x, weight, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


class TestRMSNorm(unittest.TestCase):
    def test_shape(self):
        x = np.random.randn(3, 5, 8)
        w = np.ones(8)
        y = rms_norm(x, w)
        self.assertEqual(y.shape, x.shape)

    def test_unit_weight(self):
        """weight=1 时, 输出最后一维的均方根应为 1."""
        np.random.seed(0)
        x = np.random.randn(2, 4, 16) * 3
        w = np.ones(16)
        y = rms_norm(x, w)
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-4)

    def test_scaling(self):
        """weight=2 时, 输出应为 weight=1 时 * 2."""
        x = np.random.randn(2, 4, 8)
        y1 = rms_norm(x, np.ones(8))
        y2 = rms_norm(x, np.ones(8) * 2)
        np.testing.assert_allclose(y2, y1 * 2, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
