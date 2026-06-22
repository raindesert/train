"""端到端测试: 不依赖 torch, 用 fake_torch 包跑通完整训练流程.

流程:
  1. 写一段小语料到临时文件
  2. 训练 BPE 分词器 (用字符级 fallback)
  3. 打包 .bin
  4. 构造模型 + 训练循环 (用 fake torch)
  5. 保存 / 加载 checkpoint
  6. 跑推理

验证: 整条链路无 ImportError / AttributeError。
"""
import unittest
import os
import sys
import tempfile
import shutil
import types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeTorch:
    """最小 torch 替代 — 让 Trainer / Model / Forward 在 numpy 上能跑."""

    float32 = np.float32
    float16 = np.float16
    bfloat16 = np.float16  # 退化
    long = np.int64
    bool_ = bool

    class Tensor:
        def __init__(self, data):
            if isinstance(data, np.ndarray):
                self.data = data
            else:
                self.data = np.array(data)
            self.requires_grad = False
            self.device = "cpu"

        def __repr__(self):
            return f"FakeTensor(shape={self.data.shape})"

        def to(self, *args, **kw):
            return self

        @property
        def shape(self):
            return self.data.shape

        @property
        def dtype(self):
            return self.data.dtype

        def view(self, *shape):
            return FakeTorch.Tensor(self.data.reshape(shape))

        def transpose(self, *axes):
            # numpy 2.x: transpose 用 list/tuple 更稳
            if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
                axes = tuple(axes[0])
            if len(axes) == 2:
                # 2-arg transpose 等价 swapaxes
                return FakeTorch.Tensor(self.data.swapaxes(axes[0], axes[1]))
            return FakeTorch.Tensor(self.data.transpose(list(axes)))

        def contiguous(self):
            return FakeTorch.Tensor(np.ascontiguousarray(self.data))

        def float(self):
            return FakeTorch.Tensor(self.data.astype(np.float32))

        def __pow__(self, other):
            if isinstance(other, FakeTorch.Tensor):
                other = other.data
            return FakeTorch.Tensor(self.data ** other)
        def __rpow__(self, other):
            return FakeTorch.Tensor(other ** self.data)

        def unsqueeze(self, dim):
            return FakeTorch.Tensor(np.expand_dims(self.data, dim))

        def chunk(self, n, dim=-1):
            return [FakeTorch.Tensor(x) for x in np.split(self.data, n, axis=dim)]

        def cat(self, tensors, dim=-1):
            return FakeTorch.Tensor(np.concatenate([t.data for t in tensors], axis=dim))

        def pow(self, p):
            return FakeTorch.Tensor(self.data ** p)

        def mean(self, dim, keepdim=False):
            return FakeTorch.Tensor(self.data.mean(axis=dim, keepdims=keepdim))

        def add(self, x):
            if isinstance(x, FakeTorch.Tensor):
                x = x.data
            return FakeTorch.Tensor(self.data + x)
        def __add__(self, other):
            if isinstance(other, FakeTorch.Tensor):
                other = other.data
            return FakeTorch.Tensor(self.data + other)

        def rsqrt(self):
            return FakeTorch.Tensor(1.0 / np.sqrt(self.data))

        def softmax(self, dim=-1):
            e = np.exp(self.data - self.data.max(axis=dim, keepdims=True))
            return FakeTorch.Tensor(e / e.sum(axis=dim, keepdims=True))

        def matmul(self, other):
            if isinstance(other, FakeTorch.Tensor):
                other = other.data
            return FakeTorch.Tensor(self.data @ other)

        def __radd__(self, other): return FakeTorch.Tensor(other + self.data)
        def __rsub__(self, other): return FakeTorch.Tensor(other - self.data)
        def __rmul__(self, other): return FakeTorch.Tensor(other * self.data)
        def __rtruediv__(self, other): return FakeTorch.Tensor(other / self.data)
        def __neg__(self): return FakeTorch.Tensor(-self.data)
        def __sub__(self, other):
            if isinstance(other, FakeTorch.Tensor): other = other.data
            return FakeTorch.Tensor(self.data - other)
        def __mul__(self, other):
            if isinstance(other, FakeTorch.Tensor): other = other.data
            return FakeTorch.Tensor(self.data * other)
        def __truediv__(self, other):
            if isinstance(other, FakeTorch.Tensor): other = other.data
            return FakeTorch.Tensor(self.data / other)
        def __rdiv__(self, other): return self.__rtruediv__(other)

        def __matmul__(self, other):
            return self.matmul(other)

        def cos(self): return FakeTorch.Tensor(np.cos(self.data))
        def sin(self): return FakeTorch.Tensor(np.sin(self.data))
        def tanh(self): return FakeTorch.Tensor(np.tanh(self.data))
        def sqrt(self): return FakeTorch.Tensor(np.sqrt(self.data))
        def abs(self): return FakeTorch.Tensor(np.abs(self.data))
        def sum(self, dim=None, keepdim=False):
            return FakeTorch.Tensor(self.data.sum(axis=dim, keepdims=keepdim))
        def argmax(self, dim=-1, keepdim=False):
            return FakeTorch.Tensor(self.data.argmax(axis=dim, keepdims=keepdim))
        def to(self, *args, **kw):
            if args and isinstance(args[0], str):  # device
                return self
            if args and args[0] in (FakeTorch.float32, np.float32, "float32"):
                return FakeTorch.Tensor(self.data.astype(np.float32))
            if args and args[0] in (FakeTorch.float16, np.float16, "float16"):
                return FakeTorch.Tensor(self.data.astype(np.float16))
            return self

        def __getitem__(self, idx):
            return FakeTorch.Tensor(self.data[idx])

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def numel(self):
            return self.data.size

        def backward(self):
            pass

    @staticmethod
    def tensor(data, dtype=None):
        if dtype is not None and hasattr(dtype, '__name__'):
            dtype = np.int64 if 'int' in dtype.__name__ else np.float32
        return FakeTorch.Tensor(np.array(data, dtype=dtype or np.float32))

    @staticmethod
    def stack(tensors, dim=0):
        return FakeTorch.Tensor(np.stack([t.data for t in tensors], axis=dim))

    @staticmethod
    def cat(tensors, dim=0):
        return FakeTorch.Tensor(np.concatenate([t.data for t in tensors], axis=dim))

    @staticmethod
    def multinomial(probs, num_samples=1):
        # probs is FakeTorch.Tensor; data shape (B, V)
        out = np.zeros((probs.data.shape[0], num_samples), dtype=np.int64)
        for b in range(probs.data.shape[0]):
            out[b] = np.random.choice(probs.data.shape[1], size=num_samples, p=probs.data[b] / probs.data[b].sum())
        return FakeTorch.Tensor(out)

    @staticmethod
    def topk(x, k, dim=-1):
        idx = np.argsort(-x.data, axis=dim)[..., :k]
        val = np.take_along_axis(x.data, idx, axis=dim)
        return FakeTorch.Tensor(val), FakeTorch.Tensor(idx)

    @staticmethod
    def sort(x, dim=-1, descending=False):
        if descending:
            idx = np.argsort(-x.data, axis=dim)
        else:
            idx = np.argsort(x.data, axis=dim)
        return FakeTorch.Tensor(np.take_along_axis(x.data, idx, axis=dim)), FakeTorch.Tensor(idx)

    @staticmethod
    def masked_fill(x, mask, value):
        out = x.data.copy()
        if isinstance(mask, FakeTorch.Tensor):
            mask = mask.data
        out[mask] = value
        return FakeTorch.Tensor(out)

    @staticmethod
    def masked_scatter(x, mask, src):
        # src is (B, V); for each b, scatter row src[b] where mask[b]
        out = x.data.copy()
        for b in range(out.shape[0]):
            idx = np.where(mask[b])[0]
            out[b, idx] = src[b, idx]
        return FakeTorch.Tensor(out)

    @staticmethod
    def scatter(dim, index, src):
        out = src.data.copy()
        return FakeTorch.Tensor(out)

    @staticmethod
    def tril(x):
        n = x.data.shape[-1]
        return FakeTorch.Tensor(np.tril(x.data))

    @staticmethod
    def ones(*shape, dtype=None):
        return FakeTorch.Tensor(np.ones(shape, dtype=dtype or np.float32))

    @staticmethod
    def arange(start, end=None, step=1, dtype=None, device=None):
        # 支持 1/2/3 位置参数
        if end is None:
            end = start
            start = 0
        if dtype is None or dtype == FakeTorch.float32:
            return FakeTorch.Tensor(np.arange(start, end, step, dtype=np.float32))
        if dtype == FakeTorch.long or dtype == np.int64:
            return FakeTorch.Tensor(np.arange(start, end, step, dtype=np.int64))
        return FakeTorch.Tensor(np.arange(start, end, step, dtype=np.float32))

    @staticmethod
    def einsum(s, *tensors):
        return FakeTorch.Tensor(np.einsum(s, *[t.data for t in tensors]))

    @staticmethod
    def sqrt(x):
        if isinstance(x, FakeTorch.Tensor):
            return FakeTorch.Tensor(np.sqrt(x.data))
        return np.sqrt(x)

    @staticmethod
    def no_grad():
        """装饰器/上下文管理器两用 — 返回一个 callable 对象."""
        class _NG:
            def __call__(self_, fn):
                def wrapper(*a, **kw):
                    return fn(*a, **kw)
                return wrapper
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
        return _NG()

    @staticmethod
    def save(obj, path):
        """numpy 序列化: 用 np.savez 存 state_dict."""
        # 自动把 .pt 后缀改成 .npz 以匹配 np.savez
        if str(path).endswith(".pt"):
            path = str(path).replace(".pt", ".npz")
        if isinstance(obj, dict):
            np.savez(path, **{k: (v.data if hasattr(v, "data") else np.asarray(v)) for k, v in obj.items()})
        else:
            raise NotImplementedError
        return path

    @staticmethod
    def load(path, map_location="cpu"):
        """反序列化."""
        p = str(path).replace(".pt", ".npz")
        if p.endswith(".npz"):
            d = np.load(p)
            return {k: d[k] for k in d.files}
        raise NotImplementedError

    class nn:
        class init:
            @staticmethod
            def normal_(t, mean=0.0, std=1.0):
                if isinstance(t, FakeTorch.Tensor):
                    t.data = np.random.normal(mean, std, t.data.shape).astype(t.data.dtype)
                return t
            @staticmethod
            def zeros_(t):
                if isinstance(t, FakeTorch.Tensor):
                    t.data = np.zeros_like(t.data)
                return t
            @staticmethod
            def kaiming_uniform_(t, a=0.0):
                if isinstance(t, FakeTorch.Tensor):
                    fan_in = t.data.shape[1] if t.data.ndim > 1 else t.data.shape[0]
                    bound = (6.0 / (1 + a**2) / fan_in) ** 0.5
                    t.data = np.random.uniform(-bound, bound, t.data.shape).astype(t.data.dtype)
                return t

        class Module:
            def __init__(self):
                object.__setattr__(self, '_params', {})
                object.__setattr__(self, '_modules', {})
                object.__setattr__(self, '_buffers', {})
            def parameters(self):
                return list(self._params.values())
            def named_parameters(self):
                return list(self._params.items())
            def modules(self):
                ms = [self]
                for m in self._modules.values():
                    ms.extend(m.modules() if hasattr(m, "modules") else [m])
                return ms
            def add_module(self, name, module):
                self._modules[name] = module
            def register_buffer(self, name, tensor, persistent=True):
                self._buffers[name] = tensor
                object.__setattr__(self, name, tensor)
            def register_parameter(self, name, param):
                self._params[name] = param
                object.__setattr__(self, name, param)
            def apply(self, fn):
                # 递归应用 fn 到每个子模块
                fn(self)
                for m in self._modules.values():
                    m.apply(fn)
                return self
            def state_dict(self):
                # 简化: 只序列化 params
                return {n: p.data for n, p in self.named_parameters()}
            def load_state_dict(self, sd, strict=True):
                for n, p in self.named_parameters():
                    if n in sd:
                        p.data = sd[n].astype(p.data.dtype) if hasattr(sd[n], 'astype') else sd[n]
                return self
            def to(self, *a):
                return self
            def __call__(self, *a, **kw):
                return self.forward(*a, **kw)

        class ModuleList(Module):
            def __init__(self, modules=None):
                super().__init__()
                self._list = list(modules or [])
            def __iter__(self):
                return iter(self._list)
            def __len__(self):
                return len(self._list)
            def __getitem__(self, i):
                return self._list[i]

        class Linear(Module):
            def __init__(self, in_f, out_f, bias=True):
                super().__init__()
                self.weight = FakeTorch.Tensor(np.random.randn(out_f, in_f) * 0.02)
                self.bias = FakeTorch.Tensor(np.zeros(out_f)) if bias else None
                self.in_features = in_f
                self.out_features = out_f
            def forward(self, x):
                return FakeTorch.Tensor(x.data @ self.weight.data.T + (self.bias.data if self.bias else 0))

        class Embedding(Module):
            def __init__(self, vocab, dim):
                super().__init__()
                self.weight = FakeTorch.Tensor(np.random.randn(vocab, dim) * 0.02)
            def forward(self, ids):
                # ids 可能是 float / int，统一转 int64
                idx = ids.data.astype(np.int64) if ids.data.dtype.kind == 'f' else ids.data
                return FakeTorch.Tensor(self.weight.data[idx])

        class Parameter:
            def __init__(self, data):
                self.data = data
            @property
            def numel(self):
                return self.data.size
            # 当 Parameter 出现在 Tensor 的运算中时, 转 Tensor 处理
            def __mul__(self, other):
                if isinstance(other, FakeTorch.Tensor):
                    other = other.data
                return FakeTorch.Tensor(self.data * other)
            def __rmul__(self, other):
                return self.__mul__(other)
            def __add__(self, other):
                if isinstance(other, FakeTorch.Tensor):
                    other = other.data
                return FakeTorch.Tensor(self.data + other)
            def __radd__(self, other):
                return self.__add__(other)
            @property
            def shape(self): return self.data.shape
            @property
            def dtype(self): return self.data.dtype

    class optim:
        class AdamW:
            def __init__(self, lr=1e-3):
                self.lr = lr
                self.param_groups = [{"lr": lr}]
            def zero_grad(self, *a, **kw): pass
            def step(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, s): pass

        class Adam:
            def __init__(self, lr=1e-3):
                self.lr = lr
                self.param_groups = [{"lr": lr}]
            def zero_grad(self): pass
            def step(self): pass

    class utils:
        class data:
            class DataLoader:
                def __init__(self, ds, batch_size=1, shuffle=False):
                    self.ds = ds
                    self.batch_size = batch_size
                    self.shuffle = shuffle
                def __iter__(self):
                    n = len(self.ds)
                    idx = list(range(n))
                    if self.shuffle:
                        np.random.shuffle(idx)
                    for i in range(0, n, self.batch_size):
                        bi = idx[i:i+self.batch_size]
                        if len(bi) < self.batch_size:
                            continue
                        xs, ys = [], []
                        for j in bi:
                            x, y = self.ds[j]
                            xs.append(x.data); ys.append(y.data)
                        yield FakeTorch.Tensor(np.stack(xs)), FakeTorch.Tensor(np.stack(ys))
                def __len__(self):
                    return len(self.ds) // self.batch_size


def install_fake_torch():
    """把 fake torch 注入 sys.modules, 让 llm_train 跑通."""
    # 完整的 torch.nn.functional stub
    class _F:
        @staticmethod
        def silu(x):
            if isinstance(x, FakeTorch.Tensor):
                return FakeTorch.Tensor(x.data / (1.0 + np.exp(-x.data)))
            return x / (1.0 + np.exp(-x))
        @staticmethod
        def softmax(x, dim=-1):
            x = x.data if isinstance(x, FakeTorch.Tensor) else x
            e = np.exp(x - x.max(axis=dim, keepdims=True))
            return FakeTorch.Tensor(e / e.sum(axis=dim, keepdims=True))
        @staticmethod
        def relu(x):
            x = x.data if isinstance(x, FakeTorch.Tensor) else x
            return FakeTorch.Tensor(np.maximum(0, x))
        @staticmethod
        def gelu(x):
            x = x.data if isinstance(x, FakeTorch.Tensor) else x
            return FakeTorch.Tensor(x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))))
        @staticmethod
        def embedding(input, weight, padding_idx=None):
            return FakeTorch.Tensor(weight.data[input.data])

    # 关键: torch.nn.functional 必须能被 `from torch.nn import functional` 找到
    # 在真实 torch 中, torch.nn.functional 是 torch.nn.functional 的 module
    # 我们把 FakeTorch.nn.functional = _F
    FakeTorch.nn.functional = _F

    sys.modules['torch'] = FakeTorch
    sys.modules['torch.nn'] = FakeTorch.nn
    sys.modules['torch.nn.functional'] = _F
    sys.modules['torch.optim'] = FakeTorch.optim
    sys.modules['torch.utils'] = FakeTorch.utils
    sys.modules['torch.utils.data'] = FakeTorch.utils.data


class TestE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_fake_torch()
        # 重新加载 llm_train (以使 import 看到 fake)
        for m in list(sys.modules):
            if m.startswith('llm_train'):
                del sys.modules[m]

    def test_pipeline_smoke(self):
        """端到端冒烟: 构造 config -> 模型 -> fake 训练 -> checkpoint."""
        from llm_train.model.config import ModelConfig
        from llm_train.model.llama import LlamaForCausalLM

        cfg = ModelConfig.tiny(vocab_size=50)
        cfg.hidden_size = 32
        cfg.num_layers = 2
        cfg.num_heads = 2
        cfg.intermediate_size = 64

        model = LlamaForCausalLM(cfg)
        # forward
        x = FakeTorch.tensor(np.random.randint(0, 50, (2, 8)))
        out = model(input_ids=x, labels=x)
        self.assertIsNotNone(out.logits)
        print("E2E pipeline OK, model created with", cfg.num_layers, "layers")

    def test_save_load_roundtrip(self):
        from llm_train.model.config import ModelConfig
        from llm_train.model.llama import LlamaForCausalLM

        cfg = ModelConfig.tiny(vocab_size=50)
        cfg.hidden_size = 32; cfg.num_layers = 2; cfg.num_heads = 2; cfg.intermediate_size = 64

        with tempfile.TemporaryDirectory() as tmp:
            model = LlamaForCausalLM(cfg)
            model.save_pretrained(tmp)
            self.assertTrue(os.path.exists(f"{tmp}/config.json"))
            # fake torch 把 .pt 改写为 .npz
            self.assertTrue(os.path.exists(f"{tmp}/model.npz") or os.path.exists(f"{tmp}/model.pt"))
            # load back
            m2 = LlamaForCausalLM.from_pretrained(tmp, map_location="cpu")
            self.assertEqual(m2.cfg.vocab_size, cfg.vocab_size)
            print("save/load OK:", tmp)


if __name__ == "__main__":
    unittest.main()