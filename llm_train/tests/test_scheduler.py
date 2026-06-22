"""LR scheduler 数学测试 (numpy)."""
import unittest
import math


def cosine_lr(step, total, warmup, base_lr=1.0, min_ratio=0.1):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return base_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t)))


class TestScheduler(unittest.TestCase):
    def test_warmup(self):
        # warmup 阶段应单调上升
        prev = -1
        for s in range(5):
            lr = cosine_lr(s, total=100, warmup=5, base_lr=1.0)
            self.assertGreater(lr, prev)
            prev = lr

    def test_cosine_decay(self):
        # 训练末期应接近 min_ratio * base
        lr_end = cosine_lr(99, total=100, warmup=5, base_lr=1.0, min_ratio=0.1)
        self.assertAlmostEqual(lr_end, 0.1, places=2)

    def test_peak_at_warmup_end(self):
        lr_at_warmup = cosine_lr(5, total=100, warmup=5, base_lr=1.0)
        # 紧接着衰减
        lr_after = cosine_lr(6, total=100, warmup=5, base_lr=1.0)
        self.assertGreaterEqual(lr_at_warmup, lr_after)


if __name__ == "__main__":
    unittest.main()
