---
title: 修测试套件 22 个错误 (encoding + Windows 文件锁)
date: 2026-06-23
---

# 目标

让 `python -m unittest discover -s llm_train/tests` 全绿。当前 50 tests / 28 pass / 2 skipped / 20~22 errors,全集中在两个测试文件,无生产代码改动。

# 背景

## Bug A: GBK encoding (20 errors,test_trainer_config.py)

`test_trainer_config.py` 通过 AST 直接读 `training/trainer.py` 源文件,目的是在无 torch 环境也能跑 from_dict 测试。

```python
# llm_train/tests/test_trainer_config.py:29
with open(TRAINER_PATH) as f:
    src = f.read()
```

`trainer.py` 头部是 UTF-8 中文 docstring (`"""通用 Trainer — 预训练 / 微调共用..."""`)。Windows 上 `open()` 不传 `encoding` 默认走 GBK,碰到 `—`(U+2014)和中文直接抛 `UnicodeDecodeError`。Linux/macOS 默认 UTF-8 所以偶发才暴露,Windows CI 必坏。

## Bug B: Windows 文件锁 (2 errors,test_packing.py)

```python
# llm_train/tests/test_packing.py:14-23
with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
    path = f.name
try:
    pack_bin(path, tokens, vocab_size=60000)
    arr = load_bin(path)          # np.memmap,持有底层 mmap/文件句柄
    self.assertEqual(arr.dtype, np.uint16)
    ...
finally:
    os.unlink(path)               # Windows: 文件仍被 mmap 占用 → PermissionError
```

`load_bin` 内部用 `np.memmap(..., mode="r")`,返回的对象绑了一个 `mmap.mmap` 实例,句柄直到对象被 GC 才会释放。Linux 上 unlink 即便 mmap 打开也能成功(unlink 后仍可读),Windows 必须先关掉句柄。

# 改动

只改测试文件,生产代码 0 改动。

## 文件 1: `llm_train/tests/test_trainer_config.py`

```diff
-    with open(TRAINER_PATH) as f:
+    with open(TRAINER_PATH, encoding="utf-8") as f:
         src = f.read()
```

## 文件 2: `llm_train/tests/test_packing.py`

两个 roundtrip 测试都改:在断言后、unlink 前显式关闭 memmap。

```diff
 def test_roundtrip_uint16(self):
     tokens = list(range(0, 50000))
     with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
         path = f.name
     try:
         pack_bin(path, tokens, vocab_size=60000)
         arr = load_bin(path)
-        self.assertEqual(arr.dtype, np.uint16)
-        self.assertEqual(arr.shape, (50000,))
-        np.testing.assert_array_equal(arr, np.array(tokens, dtype=np.uint16))
+        try:
+            self.assertEqual(arr.dtype, np.uint16)
+            self.assertEqual(arr.shape, (50000,))
+            np.testing.assert_array_equal(arr, np.array(tokens, dtype=np.uint16))
+        finally:
+            # Drop mmap reference before unlink (Windows file lock)
+            if hasattr(arr, "_mmap") and arr._mmap is not None:
+                arr._mmap.close()
     finally:
         os.unlink(path)
```

`test_roundtrip_uint32` 套同样的 pattern。

`test_magic_header` 不走 memmap (只用 `open(path, "rb")`),已经 OK,不动。

# 不做

- 不重写测试结构 (TemporaryDirectory / pytest tmp_path fixture) — 收益低、改动面大
- 不动 `packing.py` 的 `load_bin` 加 context manager 支持 — 当前 API 没问题,只在测试侧补 cleanup
- 不补缺失模块测试 — 下一轮范围

# 验证

```bash
python -m unittest discover -s llm_train/tests -v
```

预期: **50 tests, 0 errors, 0 failures**,skip 数仍为 2。

单独跑 packing 测试:

```bash
python -m unittest llm_train.tests.test_packing -v
```

预期: 3 tests 全 pass (含 magic_header)。