""".bin 打包测试."""
import unittest
import os, tempfile, struct
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_train.data.packing import pack_bin, load_bin, MAGIC, VERSION
import numpy as np


class TestPacking(unittest.TestCase):
    def test_roundtrip_uint16(self):
        tokens = list(range(0, 50000))
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            pack_bin(path, tokens, vocab_size=60000)
            arr = load_bin(path)
            try:
                self.assertEqual(arr.dtype, np.uint16)
                self.assertEqual(arr.shape, (50000,))
                np.testing.assert_array_equal(arr, np.array(tokens, dtype=np.uint16))
            finally:
                # Release mmap before unlink (Windows file lock)
                if hasattr(arr, "_mmap") and arr._mmap is not None:
                    arr._mmap.close()
        finally:
            os.unlink(path)

    def test_roundtrip_uint32(self):
        # vocab_size > 65535 -> uint32
        tokens = list(range(70000))
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            pack_bin(path, tokens, vocab_size=100000)
            arr = load_bin(path)
            try:
                self.assertEqual(arr.dtype, np.uint32)
                np.testing.assert_array_equal(arr, np.array(tokens, dtype=np.uint32))
            finally:
                # Release mmap before unlink (Windows file lock)
                if hasattr(arr, "_mmap") and arr._mmap is not None:
                    arr._mmap.close()
        finally:
            os.unlink(path)

    def test_magic_header(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            pack_bin(path, [1, 2, 3], vocab_size=100)
            with open(path, "rb") as f:
                magic = f.read(4)
                ver, vocab = struct.unpack("<II", f.read(8))
            self.assertEqual(magic, MAGIC)
            self.assertEqual(ver, VERSION)
            self.assertEqual(vocab, 100)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
