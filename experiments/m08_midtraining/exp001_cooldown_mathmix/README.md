# M8：213M Midtraining、Cooldown 与数学数据混合

## 结论

从同一 M4 checkpoint 追加约 100M tokens 时，纯 FineWeb-Edu 的 LR 从
`5e-5` 退火到 `5e-6`，比固定 `5e-5` 的新 FineWeb 验证 loss 低
**0.013465**，M4 通用验证 loss 低 **0.015222**。在相同 cooldown 和
token 预算下，把一半训练序列换成 OpenWebMath，独立数学文本验证 loss
从 **3.724689** 降到 **2.695273**；代价是新、旧 FineWeb 验证 loss
分别上升 **0.016129**、**0.015655**。两项通用零样本选择题没有显示
可靠的能力差异。现有文件中没有 MathQA 评测结果，因此不能从数学文本
loss 推断数学解题能力提升；M8 不继续追加数据配比或预算搜索。
本次只测一个追加预算，不能据此外推最优阶段 token 数。

## 设置与结果

共同起点是 `runs/m04_213m_lr5e4/checkpoint.pt`：213,156,864 参数，
已训练 999,999,488 输入 tokens。阶段初始化保留模型、AdamW 动量和各
rank RNG，重置数据游标与 LR 计划。三组均为双卡 BF16 DDP、每卡
batch=4、累积=2、全局 batch=16、seed=2026、序列长 2048、无 warmup、
梯度裁剪 1.0；每组训练 **48,828** 条序列、**99,999,744** 输入
tokens、**3052** 次更新。前两组使用同一份新 FineWeb 数据和顺序，
仅 LR 计划不同；数学混合组沿用 cooldown，仅改变数据。

`m08_fineweb_100m` 的训练/验证分别为 48,828/488 条，使用 M4 的
Mistral 32K tokenizer，按文档 SHA-256 排除 M4 已用文档；本次新数据
实际来自一个 FineWeb-Edu Parquet 分片，故另用 M4 验证集检查通用表现。
`m08_math50_100m` 从新 FineWeb 和 OpenWebMath 各取 **24,414** 条
完整序列、随机混排，独立数学验证集为 488 条序列。数学文档与 M4、
M8 FineWeb 文档，以及数学训练/验证之间的正文哈希交集均为 **0**；
混合来源、序列数及抽样/混排哈希见数据 `summary.json`。

| checkpoint（`runs/` 下） | LR 计划 | 新 FineWeb loss ↓ | M4 通用 loss ↓ | 数学文本 loss ↓ | HellaSwag / ARC-Easy `acc_norm` ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| M4 起点 `m04_213m_lr5e4` | — | 2.898978 | 2.878212 | 3.756355 | 29.56% / 43.10% |
| `m08_213m_constant` | 固定 `5e-5` | 2.883971 | 2.870252 | 未测 | 29.79% / 42.30% |
| `m08_213m_cooldown` | `5e-5 → 5e-6` | **2.870506** | **2.855030** | 3.724689 | 29.68% / 42.38% |
| `m08_213m_math50_cooldown` | `5e-5 → 5e-6` | 2.886635 | 2.870685 | **2.695273** | 29.64% / 42.89% |

新 FineWeb、数学验证各有 **998,936** 个预测位置；M4 通用验证有
**9,993,454** 个。不同验证集的 loss 不能横向比较，只能在同一列中
比较 checkpoint；这些验证集参与了方案比较，不是独立测试集。选择题使用
lm-evaluation-harness 0.4.13、0-shot、
完整任务；HellaSwag 和 ARC-Easy 各为 10,042/2,376 题。表中选择题
差值小于各任务约 0.46/1.02 个百分点的单组标准误，不能据此认定
通用能力改善或遗忘。数学混合组相对 M4 起点的两份 FineWeb loss 仍低
**0.012344**、**0.007527**，但相对纯 FineWeb cooldown 明显回退。

数学混合组有 **78/3052** 次裁剪前梯度范数超过 1，最大 **13.23**；
纯 FineWeb cooldown 仅 **1/3052** 次。数学组的数学验证 loss 从
3.756355 持续降至 2.695273，末步梯度范数约 0.702，未见持续发散。
三组第 301–3000 步训练吞吐中位数分别约为 **92.8k、92.7k、93.4k**
全局 tokens/s，每卡 PyTorch allocated 峰值均约 **15.2 GiB**。

## 复现与产物

以下在项目根目录运行；现有输出目录已占用，重跑时改用新目录。
FineWeb 输入目录是本次选出的原始分片，复现数据顺序须保留同一文件。

```bash
PYTHONNOUSERSITE=1 python scripts/data/prepare_fineweb.py \
  --input-dir /tmp/dummym_m08_source \
  --tokenizer data/tokenized/m04_fineweb_1b/tokenizer.json \
  --exclude-documents data/tokenized/m04_fineweb_1b/documents.jsonl \
  --train-tokens 100000000 --validation-tokens 1000000 \
  --output-dir data/tokenized/m08_fineweb_100m

PYTHONNOUSERSITE=1 python scripts/data/prepare_m08_math_mix.py \
  --math-input-dir /diff/workspace/wyl/data/open-web-math \
  --fineweb-data-dir data/tokenized/m08_fineweb_100m \
  --exclude-documents data/tokenized/m04_fineweb_1b/documents.jsonl \
  --exclude-documents data/tokenized/m08_fineweb_100m/documents.jsonl \
  --output-dir data/tokenized/m08_math50_100m

COMMON=(--model-config configs/model/p213m.yaml \
  --init-checkpoint runs/m04_213m_lr5e4/checkpoint.pt \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 \
  --epochs 1 --learning-rate 5e-5 --warmup-steps 0 --seed 2026 \
  --eval-every 500 --save-every 1000 --log-every 100)
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py "${COMMON[@]}" \
  --data-dir data/tokenized/m08_fineweb_100m --min-lr-ratio 1 \
  --output-dir runs/m08_213m_constant
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py "${COMMON[@]}" \
  --data-dir data/tokenized/m08_fineweb_100m --min-lr-ratio 0.1 \
  --output-dir runs/m08_213m_cooldown
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py "${COMMON[@]}" \
  --data-dir data/tokenized/m08_math50_100m --min-lr-ratio 0.1 \
  --output-dir runs/m08_213m_math50_cooldown

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python scripts/eval/base_eval.py loss \
  --checkpoint runs/m08_213m_math50_cooldown/checkpoint.pt \
  --data-dir data/tokenized/m08_math50_100m --device cuda:0 --precision bf16 \
  --output runs/m08_eval/math50_math.json
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python scripts/eval/base_eval.py benchmark \
  --checkpoint runs/m08_213m_math50_cooldown/checkpoint.pt \
  --device cuda:0 --precision bf16 --tasks hellaswag,arc_easy --limit 0 \
  --output runs/m08_eval/benchmark_math50.json
```

训练摘要、checkpoint 和 TensorBoard 在各 `runs/m08_213m_*/` 目录；
其余独立 loss 评测更换 checkpoint、数据目录与输出名。评测 JSON 在
`runs/m08_eval/`，M4 基线在
`runs/m04_eval/`。截至本报告，`runs/m08_eval/` 没有 MathQA JSON；
若另有已完成的结果，需核对 checkpoint、任务及全量评测口径后补入。

## 后续

按项目路线进入 M9：先定义 chat template，完成 SFT/偏好/可验证奖励
数据审计及独立评测划分；只跑一条 SFT 基线，再从同一个 SFT checkpoint
分别进行 DPO 和 GRPO/RLVR 对照。
