# M5：39M、99M、213M 等算力对照与留出预算验证

## 结论

以 `6 × 参数量 N × 训练输入 tokens D` 作为近似计算量，在约
`1.279e17` 的预算内，39M / 547M token 的验证 loss **3.272709**，低于
99M / 215M 的 **3.360745** 和 213M / 100M 的 **3.575248**；在约
`5.935e17` 的预算内，99M / 1B 的 **3.010837** 略低于 213M / 464M 的
**3.057054**。两档的最佳结果均位于已测规模的左边界，不能据此确定真正的
计算最优模型大小，也不能仅凭这些点外推 1.15B ladder。

seed=2027 的有效复跑中，39M / 547M 和 99M / 215M 的验证 loss 分别为
**3.292105** 和 **3.362054**。39M 在两个 seed 下都更低，差值分别为
0.088036、0.069949；这是低预算的局部结论，不足以确定最优规模。
预先冻结的 `2.00e17` 留出预算中，39M / 856M 的 loss **3.209490**，
低于 99M / 337M 的 **3.239907**，预测排序正确。该点是已见模型规模内的
预算插值，不构成向 1.15B 模型外推成功的证据。

## 设置与结果

主对照全部使用同一 tokenizer、序列长度 2048、双卡 BF16 DDP、全局 batch=16、
seed=2026、AdamW、warmup=300、cosine 最低 LR 比例 0.1。39M 和 99M 使用
LR=1e-3，213M 使用 LR=5e-4；这是按已观察训练情况选的模型级 recipe，
并非固定 LR 的纯架构消融。`--train-sequences` 先取 1B `train.bin` 前 N 条，
仅在选定范围内按 seed 打乱，同时重算完整步数和 LR 计划；没有重复使用数据。
M1 的 100M `train.bin` 是 1B 训练文件的字节前缀。

所有 loss 均在同一个 4,882 序列、9,993,454 预测位置的验证集上，以 BF16
评测；HellaSwag / ARC-Easy 为 lm-evaluation-harness 0.4.13、完整任务、
0-shot 的 `acc_norm`。100M 训练原本使用约 1M-token 的内部验证集，表中改用
独立的 10M-token 同集评测，不能混用两个口径。

| 对照与产物目录 | 参数量 | 训练 tokens | 近似 `6ND` | 同集 loss ↓ | HellaSwag ↑ | ARC-Easy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定 token：`m03_99m_ddp` | 98.91M | 100.00M | 0.059e18 | 3.627178 | 26.62% | 33.08% |
| 低预算：`m05_39m_547m_lr1e-3` | 38.94M | 547.42M | 0.128e18 | **3.272709** | 未测 | 未测 |
| 低预算：`m05_99m_215m` | 98.91M | 215.48M | 0.128e18 | 3.360745 | 26.50% | 36.53% |
| 低预算：`m05_213m_100m` | 213.16M | 100.00M | 0.128e18 | 3.575248 | 26.48% | 33.84% |
| 高预算：`m05_99m_1b` | 98.91M | 1000.00M | 0.593e18 | **3.010837** | 28.27% | 40.70% |
| 高预算：`m05_213m_464m` | 213.16M | 464.03M | 0.593e18 | 3.057054 | 27.91% | 40.70% |
| 固定 token：`m04_213m_lr5e4` | 213.16M | 1000.00M | 1.279e18 | 2.878212 | 29.56% | 43.10% |

固定 100M tokens 时，213M 比 99M 的 loss 低 0.051930；固定 1B tokens
时低 0.132625。这是固定数据量的比较，不是等计算量的比较。低预算等算力
对照中，39M 比 99M 低 0.088036，99M 比 213M 低 0.214503；高预算中
99M 比 213M 低 0.046217。高预算的 ARC-Easy 两组均为 40.70%，
HellaSwag 仅差 0.36 个百分点，不能把 loss 差异直接称作能力优势。

39M 的 547M-token 运行先在同一完整 16,706 步 LR 计划下试跑 4,000 步：
`5e-4 / 1e-3` 的验证 loss 为 3.751449 / **3.711125**。只将 `1e-3`
从 checkpoint 继续至完整预算，最终 loss **3.272709**，perplexity
**26.3827**；另一组仍为 `paused`，不能拿 4,000 步结果与完整结果比较。
该 LR 选择只是短跑筛选，不证明它在完整预算下最优。

三组新等算力训练的验证 loss 一直下降，孤立的梯度尖峰在下一步恢复，没有
持续发散。`6ND` 未精确计入注意力和硬件效率：低预算三组实际训练会话合计约
1,871 秒（39M，含两次会话）、1,503 秒（99M）、1,166 秒（213M），
所以“等近似 FLOPs”不等于“等耗时”。同一验证集也参与 LR 和模型选择，
不是独立测试集。低预算最接近的两组已有配对 seed，但高预算只有单 seed；
不同模型 LR 与目前的边界最优结果仍限制了 scaling law 结论。

### seed=2027 复核

| 运行 | `--train-sequences` | 实际步数 | 实际输入 tokens | 验证 loss | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| `m05_39m_547m_seed2027` | 267296 | 16706 | 547422208 | 3.292105 | 有效复跑 |
| `m05_99m_215m_seed2027_full` | 105216 | 6576 | 215482368 | 3.362054 | 有效复跑 |
| `m05_99m_215m_seed2027` | **10521** | 658 | 21547008 | 4.717641 | 预算错误，排除等算力对照 |

39M 两个 seed 的验证 loss 为 3.272709 / 3.292105，相差 0.019397；
99M 为 3.360745 / 3.362054，相差 0.001309。四条有效验证曲线均下降至终点，
99M 的两条曲线也基本重合。两个 seed 的低预算排序一致，但只用两个 seed
不能给出可靠置信区间。误设 10521 条的运行同时缩短了训练和 LR 计划，
不能与完整预算比较，也不能直接延长该 checkpoint。
39M 的 seed=2027 在第 508 步出现一次梯度范数 15.74，随后两步为
1.04、0.74；训练使用 `max_grad_norm=1.0`，验证曲线没有持续恶化。

探索性检查：用除 `m04_213m_lr5e4` 外的六个 seed=2026 点，拟合
`L=E+A(N/100M)^(-α)+B(D/100M)^(-β)`（SciPy 有界最小二乘），得到
`E=2.461、A=0.212、α=0.599、B=0.967、β=0.466`。对留出的 213M / 1B 点预测 loss
2.927，实际为 2.878，误差 0.049；逐一删去六个拟合点重新拟合时，该预测在
2.828–2.959 间变动。六点拟合五个参数且训练结果已被观察过，这只是回顾性
稳健性检查，不能当作已验证的 scaling law 或可靠置信区间。

下一次留出预算预先定为近似 `6ND=2.00e17`，在当前 1B-token 数据容量内。
以两组重复 seed 的平均 loss 替换其原始单点、其他四点不变，按同一形式拟合，
对 39M / 856.064M tokens 和 99M / 336.986M tokens 的预测分别为约
**3.200、3.225**。逐一删去六个配置点后的预测范围分别为
3.149–3.226、3.171–3.238，彼此重叠；因此这只是事前冻结的探索性预测，
新预算结果用于检查其误差，不能因预测 39M 略优就宣称确定了最优规模。

### `2.00e17` 留出预算结果

| 运行 | 输入 tokens | 步数 | 冻结预测 loss | 实测 loss | 实测减预测 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m05_holdout_2e17_39m_seed2026` | 856064000 | 26125 | 3.200 | **3.209490** | +0.009490 |
| `m05_holdout_2e17_99m_seed2026` | 336986112 | 10284 | 3.225 | 3.239907 | +0.014907 |

两组均完成 1 epoch，数据、tokenizer、验证集和训练 recipe 与原实验一致；
近似 `6ND` 相差约 0.001%。验证 loss 均持续下降，39M 比 99M 低
**0.030418**。两组实测 loss 都比冻结预测略高，预测的排序和量级与结果
吻合，但只有一个留出预算、一个 seed，不能据此证明全局 scaling law。
39M / 99M 训练会话分别耗时约 2914 / 2307 秒，近似等 FLOPs 不等于等耗时。
组间差值超过事先约定的 0.02 实用界线，不补第二个 seed。

## 复现与产物

在项目根目录运行。下表给出每个新训练所需的模型、数据、`--train-sequences`
和 LR；100M 组读取 M1 数据，不传该参数。全局其余参数与下方命令相同。
已有 `runs/` 目录不能覆盖，重跑时须更换 `--output-dir`。

| 运行 | 模型配置 | 数据目录 | `--train-sequences` | LR | 更新步数 |
| --- | --- | --- | ---: | ---: | ---: |
| `m05_213m_100m` | `configs/model/p213m.yaml` | `data/tokenized/m01_fineweb_100m` | 不传 | 5e-4 | 3052 |
| `m05_99m_1b` | `configs/model/p099m.yaml` | `data/tokenized/m04_fineweb_1b` | 不传 | 1e-3 | 30518 |
| `m05_99m_215m` | `configs/model/p099m.yaml` | `data/tokenized/m04_fineweb_1b` | 105216 | 1e-3 | 6576 |
| `m05_213m_464m` | `configs/model/p213m.yaml` | `data/tokenized/m04_fineweb_1b` | 226576 | 5e-4 | 14161 |
| `m05_39m_547m_lr1e-3` | `configs/model/ladder/v001/p039m.yaml` | `data/tokenized/m04_fineweb_1b` | 267296 | 1e-3 | 16706 |

以最后一组为例，正式训练先在第 4000 步暂停，再以完全相同的训练参数恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py --data-dir data/tokenized/m04_fineweb_1b \
  --model-config configs/model/ladder/v001/p039m.yaml \
  --output-dir runs/m05_39m_547m_lr1e-3 --train-sequences 267296 \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --learning-rate 1e-3 --warmup-steps 300 --min-lr-ratio 0.1 --seed 2026 \
  --eval-every 1000 --save-every 2000 --stop-after-steps 4000

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py --data-dir data/tokenized/m04_fineweb_1b \
  --model-config configs/model/ladder/v001/p039m.yaml \
  --resume runs/m05_39m_547m_lr1e-3/checkpoint.pt --train-sequences 267296 \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --learning-rate 1e-3 --warmup-steps 300 --min-lr-ratio 0.1 --seed 2026 \
  --eval-every 1000 --save-every 2000
```

训练 checkpoint、summary 和曲线留在各自的 `runs/` 目录；同集独立评测
位于 `runs/m04_eval/`、`runs/m05_eval/`。使用 M4 的 10M-token 数据训练的
各组已在训练过程中运行全量验证，无需重复执行 loss 评测。

## 留出预算复现

两组从头训练，不能用旧 checkpoint 继续，因为改变总步数会改变 cosine
LR 计划。以下为已完成运行的命令；事先设定的第二 seed 复核界线为组间 loss
差值不超过 0.02，它不是统计置信区间。

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py --data-dir data/tokenized/m04_fineweb_1b \
  --model-config configs/model/ladder/v001/p039m.yaml \
  --output-dir runs/m05_holdout_2e17_39m_seed2026 --train-sequences 418000 \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --learning-rate 1e-3 --warmup-steps 300 --min-lr-ratio 0.1 --seed 2026 \
  --eval-every 1000 --save-every 2000

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py --data-dir data/tokenized/m04_fineweb_1b \
  --model-config configs/model/p099m.yaml \
  --output-dir runs/m05_holdout_2e17_99m_seed2026 --train-sequences 164544 \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --learning-rate 1e-3 --warmup-steps 300 --min-lr-ratio 0.1 --seed 2026 \
  --eval-every 1000 --save-every 2000
```

## 后续

本轮 M5 到此收束：保留 39M 在两档低预算中的局部优势、留出预算预测误差，
以及高预算只比较过 99M/213M 的限制；不再追加 M5 训练或宣称已找到计算
最优规模。按项目路线进入 M6 优化器实验。先在现有 99M / 100M-token
DDP 基线上实现可切换的 Muon：AdamW 基线为 `runs/m03_99m_ddp`，
其独立 10M-token 验证 loss 为 3.627178。保持数据、模型、batch、seed 与
验证方式一致，对新优化器另选学习率；不重复已有 AdamW 基线。先验证
短跑的数值和恢复，再运行完整预算并比较 loss、吞吐、参数范数与
update/parameter ratio。
