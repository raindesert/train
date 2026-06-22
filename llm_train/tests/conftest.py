"""跳过 torch 时的 fallback.

如果 torch 没装, 用 numpy 构造一个最小 tensor-like stub。
本测试套件的目标:
  1. 不依赖 torch 也能跑 (语法 + 形状 + 数学正确性, 用 numpy 模拟)
  2. 有 torch 时再跑一次真测试 (via pytest.mark.torch)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
