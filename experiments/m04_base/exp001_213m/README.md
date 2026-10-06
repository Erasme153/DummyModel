# M4：213M Base Model 预训练

## 结论

213,156,864 参数模型已用双卡 DDP 完成 1B-token 预算（30,518 次更新）。同一份
9,993,454 个预测位置的独立验证集上，loss **2.878212**、perplexity **17.7824**；
HellaSwag / ARC-Easy 的零样本 `acc_norm` 为 **29.56% / 43.10%**。这是当前
213M base checkpoint 的成绩，不代表具备指令遵循或可靠事实回答能力。

## 设置与学习率选择

数据为 `data/tokenized/m04_fineweb_1b`：488,281 条、999,999,488 个训练输入
tokens；4,882 条、9,998,336 个验证输入 tokens。序列长 2048，沿用 M1 的
Mistral 32K tokenizer。模型使用 `configs/model/p213m.yaml`，双卡 BF16 DDP，
每卡 batch=4、累积=2、全局 batch=16；AdamW、weight decay=0.1、
betas=(0.9,0.95)、梯度裁剪阈值 1.0、seed=2026、warmup=300、最低 LR 比例 0.1。
从头对比 `1e-3` 和 `5e-4`，两组都按完整 30,518 步余弦计划运行，只在第
6000 步暂停比较。

| 步数 | LR=1e-3 验证 loss | LR=5e-4 验证 loss |
| ---: | ---: | ---: |
| 1000 | 4.252201 | 4.283308 |
| 3000 | 3.670050 | 3.645487 |
| 6000 | 3.427497 | **3.393337** |

第 1000 步 `1e-3` 略好，之后 `5e-4` 反超，因此只将后者恢复至完整预算。
`1e-3` 的 6000 步结果不能与 `5e-4` 的最终结果比较，也未证明 `5e-4`
是该模型的全局最优 LR。完整训练的验证 loss 从第 6000 步的 3.393337 降至
**2.878212**，最后 3000 步仍下降；没有观察到持续发散。记录的裁剪前
gradient norm 大于 1 的更新为 78 / 30,518 次，最后一步为 0.853。

| 完整预算产物 | 结果 |
| --- | ---: |
| 训练输入 tokens / 更新步数 | 999,999,488 / 30,518 |
| 独立验证 loss / perplexity | 2.878212 / 17.7824 |
| HellaSwag `acc_norm`，0-shot / 样本数 | 29.56% / 10,042 |
| ARC-Easy `acc_norm`，0-shot / 样本数 | 43.10% / 2,376 |
| 双卡全局稳态训练吞吐 / 每卡 PyTorch allocated 峰值 | 约 93,100 tokens/s / 14.67 GiB |

独立 loss 评测使用全量验证集，SHA256 为
`d9326062996c358457860eb51624f2198bec7d7f282648c18e10930024e91f9c`。
benchmark 使用 lm-evaluation-harness 0.4.13、BF16、完整任务、0-shot；
`acc_norm` 是长度归一化后的选择题准确率。训练摘要、checkpoint、TensorBoard
位于 `runs/m04_213m_lr5e4/`，独立评测位于 `runs/m04_eval/`。

## 复现命令

以下从项目根目录执行；现有输出目录已占用，重跑时须更换目录。`1e-3` 组使用
相同命令，仅将 LR 改为 `1e-3`、目录改为 `runs/m04_213m_lr1e3`，到第
6000 步停止。所选 `5e-4` 组从第 6000 步恢复时保持其余参数不变：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py --data-dir data/tokenized/m04_fineweb_1b \
  --model-config configs/model/p213m.yaml --output-dir runs/m04_213m_lr5e4 \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --learning-rate 5e-4 --warmup-steps 300 --min-lr-ratio 0.1 --seed 2026 \
  --eval-every 1000 --save-every 1000 --stop-after-steps 6000

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py --data-dir data/tokenized/m04_fineweb_1b \
  --model-config configs/model/p213m.yaml --resume runs/m04_213m_lr5e4/checkpoint.pt \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --learning-rate 5e-4 --warmup-steps 300 --min-lr-ratio 0.1 --seed 2026 \
  --eval-every 1000 --save-every 1000

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python scripts/eval/base_eval.py loss \
  --checkpoint runs/m04_213m_lr5e4/checkpoint.pt \
  --data-dir data/tokenized/m04_fineweb_1b --device cuda:0 --precision bf16 \
  --output runs/m04_eval/val_213m.json

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python scripts/eval/base_eval.py benchmark \
  --checkpoint runs/m04_213m_lr5e4/checkpoint.pt --device cuda:0 --precision bf16 \
  --tasks hellaswag,arc_easy --limit 0 --output runs/m04_eval/benchmark_213m.json
```

## 模型卡与边界

这是纯 FineWeb-Edu 英文文本训练的 decoder-only base model，输入上限 2048
tokens，未做 SFT、对齐或安全评测。Mistral tokenizer、训练语料处理与验证隔离
沿用 M1 规则；验证集也用于选择 LR，loss 不是独立测试集成绩。零样本两项
选择题不是通用语言能力的完整衡量。checkpoint 含
模型、优化器和训练进度，用现有推理入口加载；当前不作为聊天模型发布。

下一步使用相同验证集区分模型规模与训练 token 数的影响，进入 M5 等算力实验。
