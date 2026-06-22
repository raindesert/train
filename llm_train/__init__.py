"""llm_train — 从零搭建的大模型训练框架 (LLaMA 风格).

模块:
    model       - Transformer / LLaMA 架构
    tokenizer   - BPE 分词器 (基于 HuggingFace tokenizers)
    data        - 数据集加载 / DataLoader / 多源混合
    training    - 优化器 / 学习率调度 / checkpoint / 分布式
    lora        - LoRA / QLoRA 低秩适配
    inference   - KV cache 推理 / 文本生成
    eval        - 困惑度 (perplexity) 评估

设计目标:
    * 代码可读性优先,不做过度抽象
    * 单 GPU / 多 GPU / DeepSpeed 友好
    * 纯 PyTorch,不依赖 transformers 训练代码
"""
__version__ = "0.1.0"
