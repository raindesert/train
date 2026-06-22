"""SwiGLU / SiLU 测试 — 不依赖 torch."""
import unittest
import math


def silu(x):
    return x / (1.0 + math.exp(-x))


def swiglu(gate, up):
    return silu(gate) * up


class TestSwiGLU(unittest.TestCase):
    def test_silu_zero(self):
        self.assertAlmostEqual(silu(0.0), 0.0)

    def test_silu_positive(self):
        self.assertGreater(silu(1.0), 0.5)

    def test_swiglu_zero_gate(self):
        # gate=0 => silu=0 => output=0
        self.assertEqual(swiglu(0.0, 5.0), 0.0)

    def test_swiglu_zero_up(self):
        self.assertEqual(swiglu(5.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
