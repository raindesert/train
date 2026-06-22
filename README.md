# llm_train — 从零搭建大模型训练框架

LLaMA 风格的轻量训练框架: 分词器 → 预训练 → 微调 → 推理 / LoRA 全流程。

## 模块总览

```
llm_train/
├── model/         # Transformer/LLaMA 模型
├── tokenizer/     # BPE 分词器训练与编解码
├── data/          # 数据集下载/加载/打包/多源混合
├── training/      # 优化器/调度/ckpt/分布式/混合精度
├── lora/          # LoRA / QLoRA
├── inference/     # KV cache + 文本生成
├── eval/          # 困惑度评估
├── configs/       # YAML 配置 (小/中/大模型)
├── scripts/       # 一键运行脚本
└── tests/         # 单元测试
```

## 依赖

核心训练: `torch>=2.0`
分词器:    `tokenizers`
数据:      `datasets`, `tqdm`, `numpy`, `requests`
配置:      `pyyaml`
LoRA:      `peft`(可选)

## 快速开始

```bash
# 1. 安装
pip install -r requirements.txt

# 2. 下载数据 (训练时按需下载到 ./data/raw/)
python -m llm_train.data.download --dataset tinyshakespeare

# 3. 训练分词器
python -m llm_train.tokenizer.train_tokenizer \
    --input data/raw/tinyshakespeare.txt \
    --output checkpoints/tokenizer \
    --vocab_size 8000

# 4. 预训练小模型 (CPU/MPS/CUDA 都行)
python scripts/pretrain.py --config configs/tiny.yaml

# 5. 全参微调
python scripts/finetune.py --config configs/sft.yaml

# 6. LoRA 微调
python scripts/lora_train.py --config configs/lora.yaml

# 7. 推理
python scripts/infer.py --checkpoint checkpoints/tiny/best.pt \
    --tokenizer checkpoints/tokenizer --prompt "Once upon a time"
```

## 模型规模

| 配置      | 参数量   | 适用                |
|-----------|----------|---------------------|
| tiny.yaml | ~15M     | CPU/MPS 调试/教学   |
| small.yaml| ~125M    | 单卡 12G+           |
| medium.yaml| ~350M   | 单卡 24G / 多卡     |
| large.yaml| ~1.3B    | 多卡/DeepSpeed      |

## 测试

```bash
python -m unittest discover -s tests -v
```
