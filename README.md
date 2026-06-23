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
├── configs/       # YAML 配置 (小/中/大模型) — 实际在 llm_train/configs/
├── scripts/       # 一键运行脚本
└── tests/         # 单元测试
```

## 依赖

核心训练: `torch>=2.0`
分词器:    `tokenizers`
数据:      `datasets`, `tqdm`, `numpy`, `requests`
配置:      `pyyaml`
LoRA:      `peft`(可选)

## 安装

```bash
pip install -r requirements.txt
```

## 数据格式支持

| 格式 | 文件类型 | 用途 |
|------|----------|------|
| 纯文本 | `.txt` | 预训练 |
| 简单 JSONL | `{"text": "..."}` | 预训练 |
| ChatML | `{"conversations": [...]}` | 微调 |
| HuggingFace Dataset | streaming | 预训练/微调 |

ChatML 格式示例：
```json
{"conversations": [
  {"role": "user", "content": "画一张春天的海报"},
  {"role": "assistant", "content": "我理解你的需求...", "tool_calls": "..."}
]}
```

## 完整流程

### 第一步：准备数据

将数据放到 `data/raw/` 目录下。推荐用 jsonl 格式。

### 第二步：处理数据（大数据必做）

如果数据文件超过 500MB，需要分片处理避免 OOM：

```bash
python scripts/split_tokenize.py \
    --input data/raw/your_data.jsonl \
    --output data/processed/your_data \
    --lines_per_chunk 100000 \
    --vocab_size 20000 \
    --train_tokenizer
```

参数说明：
- `--lines_per_chunk`: 每个分片的行数（默认 10 万行）
- `--vocab_size`: 词表大小（默认 32000）
- `--train_tokenizer`: 是否训练新 tokenizer（首次需要）
- `--tokenizer_limit`: 训练 tokenizer 的最多行数（默认 50 万，避免 OOM）

处理后得到 `data/processed/your_data.bin`，可直接用于训练。

### 第三步：训练分词器（小数据）

如果数据量小于 500MB，可以直接训练 tokenizer：

```bash
python scripts/train_tokenizer.py \
    --input data/raw/your_data.jsonl \
    --output checkpoints/tokenizer \
    --vocab_size 20000 \
    --limit 500000
```

`--limit` 限制训练行数，避免 OOM。词表不需要全量数据。

### 第四步：预训练

修改配置文件 `llm_train/configs/your_model.yaml`，指定数据路径和 tokenizer：

```yaml
model:
  vocab_size: 20000
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  max_seq_len: 1024

data:
  seq_len: 1024
  batch_size: 16
  grad_accum: 4

tokenizer:
  path: checkpoints/tokenizer   # 直接加载已训练的 tokenizer
  vocab_size: 20000

training:
  max_steps: 10000
  warmup_steps: 500
  lr: 3e-4
  amp: true
  amp_dtype: bf16
  log_every: 50
  eval_every: 500
  eval_max_batches: 5
  save_every: 1000
  out_dir: ./checkpoints/your_model
```

启动训练（自动检测 GPU/CPU）：

```bash
python scripts/pretrain.py --config llm_train/configs/your_model.yaml
```

训练会自动从 `tokenizer.path` 加载 tokenizer。如果路径不存在，会用 `input_files` 训练一个。

断点续训：
```yaml
training:
  resume: ./checkpoints/your_model/latest.pt
```

### 第五步：微调

准备微调数据（ChatML 格式），然后：

```bash
python scripts/finetune.py \
    --config llm_train/configs/sft.yaml \
    --data data/raw/sft_data.jsonl \
    --tokenizer checkpoints/tokenizer
```

会自动检测 `conversations` 字段并使用 ChatML 格式。

### 第六步：LoRA 微调

```bash
python scripts/lora_train.py \
    --config llm_train/configs/lora.yaml \
    --data data/raw/sft_data.jsonl \
    --tokenizer checkpoints/tokenizer
```

### 第七步：推理

```bash
python scripts/infer.py \
    --checkpoint checkpoints/your_model/best.pt \
    --tokenizer checkpoints/tokenizer \
    --prompt "Once upon a time"
```

支持参数：`--max_new_tokens`, `--temperature`, `--top_k`, `--top_p`, `--no_sample`

## 模型规模

| 配置      | 参数量   | 适用                |
|-----------|----------|---------------------|
| tiny.yaml | ~15M     | CPU/MPS 调试/教学   |
| small.yaml| ~60M     | 单卡 12G+           |
| medium.yaml| ~350M   | 单卡 24G / 多卡     |
| large.yaml| ~1.3B    | 多卡/DeepSpeed      |

## 测试

```bash
python -m unittest discover -s tests -v
```