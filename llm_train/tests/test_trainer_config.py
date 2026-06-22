"""TrainerConfig.from_dict 单元测试 — 不依赖 torch。

Termux 上没有 torch, 直接 import trainer.py 会失败。
所以本测试用 AST 提取 from_dict 函数定义, 在 stub 环境下 exec 后再跑测试。

如果环境有 torch, 应该用 unittest.mock 或 conftest 的 torch stub 跑真实 import。
"""
import ast
import sys
import os
import textwrap
import warnings
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAINER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "trainer.py",
)


def _load_from_dict_and_unwrap():
    """AST 提取 TrainerConfig.from_dict 和 _unwrap_optional, exec 到 stub 命名空间。

    Returns:
        (from_dict_fn, unwrap_optional_fn)
    """
    with open(TRAINER_PATH) as f:
        src = f.read()
    tree = ast.parse(src)

    fns = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fns[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    fns[f"__{node.name}__{item.name}__"] = ast.get_source_segment(src, item)

    # TrainerConfig stub — 必须与 trainer.py 字段完全一致 (写测试时同步更新)
    stub = '''
import warnings, dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, get_type_hints

@dataclass
class TrainerConfig:
    out_dir: str = "./checkpoints/run"
    max_steps: int = 1000
    eval_every: int = 200
    save_every: int = 200
    log_every: int = 20
    grad_accum: int = 1
    grad_clip: float = 1.0
    amp: bool = False
    amp_dtype: str = "auto"
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    schedule: str = "cosine"
    seed: int = 42
    device: str = "auto"
    resume: Optional[str] = None
    max_grad_norm: float = 1.0
    eval_max_batches: Optional[int] = None
'''
    ns = {}
    exec(stub, ns)

    for name, code in fns.items():
        code = textwrap.dedent(code)
        code = "\n".join(ln for ln in code.splitlines() if "@classmethod" not in ln)
        if name == "_unwrap_optional":
            exec(code, ns)
        elif name == "__TrainerConfig__from_dict__":
            code = code.replace("cls(**clean)", "TrainerConfig(**clean)")
            code = code.replace(
                'def from_dict(cls, d: dict) -> "TrainerConfig":',
                "def from_dict(d: dict):",
            )
            code = code.replace("def from_dict(cls, d: dict):", "def from_dict(d: dict):")
            ns["cls"] = ns["TrainerConfig"]
            exec(code, ns)

    return ns["from_dict"], ns["_unwrap_optional"]


class TestTrainerConfigFromDict(unittest.TestCase):
    """TrainerConfig.from_dict 容错行为测试 (不依赖 torch)。"""

    def setUp(self):
        # 实例属性 (非 class 属性) 避免 unittest 把函数当 bound method
        self.from_dict, _ = _load_from_dict_and_unwrap()

    def _expect_warns(self, d, *, expect_unknown=0, expect_type_fail=0):
        """跑 from_dict, 返回 (cfg, unknown_warnings, type_warnings)。"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = self.from_dict(d)
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        return cfg, [m for m in msgs if "unknown field" in m], [m for m in msgs if "bad type" in m]

    def test_empty_returns_defaults(self):
        cfg, _, _ = self._expect_warns({})
        self.assertEqual(cfg.max_steps, 1000)
        self.assertEqual(cfg.lr, 3e-4)
        self.assertFalse(cfg.amp)

    def test_known_fields_no_warnings(self):
        cfg, uk, tf = self._expect_warns({"max_steps": 500, "lr": 1e-3, "amp": True})
        self.assertEqual(len(uk), 0)
        self.assertEqual(len(tf), 0)
        self.assertEqual(cfg.max_steps, 500)
        self.assertAlmostEqual(cfg.lr, 1e-3)
        self.assertTrue(cfg.amp)

    def test_unknown_field_triggers_warning(self):
        cfg, uk, _ = self._expect_warns({"batch_szie": 8, "max_steps": 100})
        self.assertEqual(len(uk), 1)
        self.assertIn("batch_szie", uk[0])
        self.assertIn("8", uk[0])
        self.assertEqual(cfg.max_steps, 100)

    def test_str_to_int_convert(self):
        cfg, _, tf = self._expect_warns({"max_steps": "500"})
        self.assertEqual(len(tf), 0)
        self.assertEqual(cfg.max_steps, 500)

    def test_bad_str_int_falls_back(self):
        cfg, _, tf = self._expect_warns({"max_steps": "abc"})
        self.assertEqual(len(tf), 1)
        self.assertIn("max_steps", tf[0])
        self.assertEqual(cfg.max_steps, 1000)  # 默认值

    def test_str_to_float(self):
        cfg, _, tf = self._expect_warns({"lr": "1e-3"})
        self.assertEqual(len(tf), 0)
        self.assertAlmostEqual(cfg.lr, 1e-3)

    def test_str_bool_true(self):
        cfg, _, tf = self._expect_warns({"amp": "true"})
        self.assertEqual(len(tf), 0)
        self.assertTrue(cfg.amp)

    def test_str_bool_no(self):
        cfg, _, tf = self._expect_warns({"amp": "no"})
        self.assertEqual(len(tf), 0)
        self.assertFalse(cfg.amp)

    def test_optional_str_none(self):
        cfg, _, _ = self._expect_warns({"resume": None})
        self.assertIsNone(cfg.resume)

    def test_optional_int_none(self):
        cfg, _, _ = self._expect_warns({"eval_max_batches": None})
        self.assertIsNone(cfg.eval_max_batches)

    def test_mixed_unknown_and_type_fail(self):
        """多个 typo + 多个 type fail 应该聚合到一次 warn (不 spam)。"""
        cfg, uk, tf = self._expect_warns({
            "batch_szie": 8,
            "max_stepz": 100,
            "max_steps": "abc",
            "lr": 1e-3,
        })
        self.assertEqual(len(uk), 1, "应该聚合为 1 个 unknown warning")
        self.assertEqual(len(tf), 1, "应该聚合为 1 个 type-fail warning")
        # 被忽略的字段和值都该出现在 warning 文本里
        self.assertIn("batch_szie", uk[0])
        self.assertIn("max_stepz", uk[0])
        # 类型错的回退到默认
        self.assertEqual(cfg.max_steps, 1000)
        # 正常的留原值
        self.assertAlmostEqual(cfg.lr, 1e-3)

    def test_full_yaml_dict(self):
        """真实场景: 完整 YAML dict 应当 0 warning。"""
        yaml_like = {
            "out_dir": "./out",
            "max_steps": 1000,
            "eval_every": 100,
            "save_every": 200,
            "log_every": 20,
            "grad_accum": 2,
            "grad_clip": 1.0,
            "amp": True,
            "amp_dtype": "bf16",
            "optimizer": "adamw",
            "lr": 3e-4,
            "weight_decay": 0.1,
            "warmup_steps": 50,
            "min_lr_ratio": 0.1,
            "schedule": "cosine",
            "seed": 42,
            "device": "cuda",
        }
        cfg, uk, tf = self._expect_warns(yaml_like)
        self.assertEqual(len(uk), 0)
        self.assertEqual(len(tf), 0)
        self.assertEqual(cfg.out_dir, "./out")
        self.assertEqual(cfg.amp_dtype, "bf16")
        self.assertEqual(cfg.device, "cuda")

    def test_resume_path_string(self):
        cfg, _, _ = self._expect_warns({"resume": "/tmp/ckpt.pt"})
        self.assertEqual(cfg.resume, "/tmp/ckpt.pt")

    def test_old_behavior_compatible(self):
        """旧调用方传 dict 没变化 — 仍然得到合法 cfg, 不会因 warn 崩溃。"""
        cfg, uk, _ = self._expect_warns({"unknown_field": 42, "max_steps": 200})
        self.assertEqual(len(uk), 1)
        self.assertEqual(cfg.max_steps, 200)

    def test_backward_compat_no_strict_keys(self):
        """from_dict 必须支持任何 dict 子集, 不强制要求任何字段。"""
        cfg, _, _ = self._expect_warns({})
        # 所有字段都该是默认值
        self.assertEqual(cfg.out_dir, "./checkpoints/run")
        self.assertEqual(cfg.optimizer, "adamw")


class TestUnwrapOptional(unittest.TestCase):
    """_unwrap_optional(Union[X, None]) -> X 测试。"""

    def setUp(self):
        _, self._unwrap = _load_from_dict_and_unwrap()

    def test_unwrap_optional_str(self):
        from typing import Optional
        self.assertIs(self._unwrap(Optional[str]), str)

    def test_unwrap_optional_int(self):
        from typing import Optional
        self.assertIs(self._unwrap(Optional[int]), int)

    def test_passthrough_plain_type(self):
        # 普通类型 (不是 Union) 原样返回
        self.assertIs(self._unwrap(str), str)
        self.assertIs(self._unwrap(int), int)


if __name__ == "__main__":
    unittest.main()
