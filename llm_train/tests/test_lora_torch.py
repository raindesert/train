"""LoRA 注入/合并测试 — 需要 torch."""
import unittest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_train.tests.conftest import HAS_TORCH
if not HAS_TORCH:
    raise unittest.SkipTest("torch not installed")

import torch
import torch.nn as nn
from llm_train.lora import apply_lora, freeze_non_lora, LoRALinear, merge_lora, save_lora, load_lora


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)


class TestLoRA(unittest.TestCase):
    def test_apply_and_freeze(self):
        m = M()
        n = apply_lora(m, r=4, alpha=8)
        self.assertEqual(n, 2)
        self.assertIsInstance(m.q_proj, LoRALinear)
        freeze_non_lora(m)
        # base 权重冻结, lora_A / lora_B 可训练
        for name, p in m.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.assertTrue(p.requires_grad, name)
            else:
                self.assertFalse(p.requires_grad, name)

    def test_zero_init_preserves_output(self):
        m = M()
        apply_lora(m, r=4, alpha=8)
        x = torch.randn(1, 8)
        out_q = m.q_proj(x)
        # 因为 B=0, LoRA 旁路为 0, 应等于 base
        base_out = m.q_proj.base(x)
        torch.testing.assert_close(out_q, base_out, atol=1e-6, rtol=1e-5)

    def test_merge_lora(self):
        m = M()
        apply_lora(m, r=4, alpha=8)
        # merge 后 q_proj 不再是 LoRALinear
        merge_lora(m)
        self.assertNotIsInstance(m.q_proj, LoRALinear)
        self.assertIsInstance(m.q_proj, nn.Linear)

    def test_save_load_lora(self):
        m = M()
        apply_lora(m, r=4, alpha=8)
        with __import__("tempfile").NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_lora(m, path)
            m2 = M()
            apply_lora(m2, r=4, alpha=8)
            load_lora(m2, path)
            for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
                if "lora_" in n1:
                    torch.testing.assert_close(p1, p2)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
