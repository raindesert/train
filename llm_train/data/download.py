"""数据集下载脚本.

维护一个预设数据集清单 (中文/英文/多模态), 通过 huggingface / github 镜像下载.
用户后续可自行添加; 数据落到 data/raw/.
"""
from __future__ import annotations
import os
import json
import urllib.request
from typing import Dict

DATA_ROOT = os.environ.get("LLM_DATA_ROOT", "./data/raw")


# 预置的几个可立即下载的小/中数据集
DATASETS: Dict[str, dict] = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "tinyshakespeare.txt",
        "desc": "莎士比亚作品拼接 (~1MB), 适合教学/CPU 调试",
    },
    "wiki103-demo": {
        "url": "https://wikitext.smerity.com/wikitext-103-v1.zip",
        "filename": "wikitext-103-v1.zip",
        "desc": "WikiText-103 (英文维基, ~500MB)",
    },
    "zhwiki-lite": {
        "url": "",   # 用户可自行下载后放到 DATA_ROOT/zhwiki-lite.txt
        "filename": "zhwiki-lite.txt",
        "desc": "中文维基 (留空, 用户需自行提供)",
    },
}


def download_file(url: str, out_path: str, chunk: int = 1 << 20) -> str:
    """下载到 out_path; 打印进度."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return out_path
    print(f"Downloading {url} -> {out_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "llm_train/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                pct = done * 100 / total
                msg = "\r  %.1fMB / %.1fMB (%.1f%%)" % (done/1e6, total/1e6, pct)
                print(msg, end="")
        print()
    return out_path


def download_dataset(name: str, root: str = DATA_ROOT) -> str:
    info = DATASETS.get(name)
    if not info:
        raise KeyError(f"未知数据集: {name}; 可选: {list(DATASETS)}")
    out = os.path.join(root, info["filename"])
    if info["url"]:
        return download_file(info["url"], out)
    print(f"[skip] {name}: 未提供 URL, 请手动放到 {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--root", default=DATA_ROOT)
    args = ap.parse_args()
    p = download_dataset(args.dataset, args.root)
    print("done:", p)
