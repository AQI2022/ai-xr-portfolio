# Qwen 小模型 LoRA 微调

实际运行 Qwen2.5-0.5B-Instruct 的参数高效监督微调，覆盖原始样本、tokenization、loss masking、反向传播、梯度裁剪、AdamW、验证、保存与重载。

```bash
python scripts/download_models.py --qwen
python experiments/finetune.py --steps 24
# 极低资源下仅验证程序，不等于预训练模型微调
python experiments/finetune.py --tiny --steps 24
```

## 参数与证据

训练资料是本仓库自编的 32 条中英文 XR/AI 问答，验证资料 8 条。两集合问题字符串无重合；部分主题相近，验证的是小规模域内适配，而不是严格跨域泛化。当前训练循环每步一个样本，24 步只使用训练列表前 24 条，未称为完整 epoch。

LoRA r=8、alpha=16、dropout=0.05，挂载 q_proj/v_proj；学习率 2e-4，seed=42，回答以外 token 的 label 为 -100。只训练 540,672 个适配器参数，总参数 494,573,440。CPU FP32 路径实际执行。

`evidence/qwen-lora.json` 保存数据 SHA256、24 步 loss、验证 token 加权 loss、困惑度、示例输出及运行时长。验证损失 4.2027 → 3.6678，适配器保存重载最大 logits 差为 0。训练曲线逐步使用不同样本，因此不能要求每个 step 的 loss 单调下降。

保留的样例回答对空间锚点的描述仍偏泛化，且输出受 48 token 限制。loss 下降说明拟合验证文本有所改善，不意味着事实质量、任务成功率或通用模型能力已提升。后续应加入人工 rubric、保留集任务评测和基座对照生成。

`--qlora` 要求 CUDA、bitsandbytes 和兼容显卡环境；本次机器的已测试 Torch 为 CPU 版本，未执行 4-bit 训练。不能把本次 LoRA 报告改称 QLoRA。

参考：[PEFT 快速入门](https://huggingface.co/docs/peft/quicktour)、[Qwen2.5-0.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。权重和适配器保存在被 Git 忽略的 models 目录，运行者需另行下载基座。
