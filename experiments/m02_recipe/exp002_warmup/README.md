# M2：99M Warmup 对比

## 结论

固定峰值 LR=1e-3，将 warmup 从 100 步增加到 300 步，在 seed=2026、2027
两组配对实验中均降低了最终验证 loss。当前采用 **LR=1e-3、warmup=300、
cosine 最低 LR=1e-4** 作为该 99M 模型、约 1 亿 token 预算下的基线。

| Seed | warmup=100 验证 loss | warmup=300 验证 loss | loss 改善量 |
| --- | ---: | ---: | ---: |
| 2026 | 3.744542 | 3.656008 | 0.088535 |
| 2027 | 3.713781 | 3.654872 | 0.058909 |
| 均值 | 3.729161 | **3.655440** | **0.073722** |

两次排名一致；warmup=300 两次结果差约 0.001136，而原设置相差约 0.030762。
这只是两个 seed 的观察，不足以证明总体方差更低或已经找到最优 schedule。

## 设置和证据

基线来自 [LR 实验报告](../exp001_lr_sweep/README.md)，复用已有 warmup=100
运行。四组均完成 1 epoch、3052 次更新、99,999,744 个输入 tokens，状态为
`complete`。每组有 3052 条训练 loss、32 个全量验证点，最终事件记录与
`summary.json` 的 step/loss 一致（允许事件文件的浮点存储误差）。

核对四组保存的 `contract`，仅 `warmup_steps` 和配对的训练 seed 不同：

- 模型：[`p099m.yaml`](../../../configs/model/p099m.yaml)，98,913,024 参数；
- 数据：`data/tokenized/m01_fineweb_100m/`，tokenizer 及 train/validation 文件指纹相同；
- 2048-token 序列，micro-batch=4，累积 4 次，有效 batch=16；
- CUDA BF16 autocast，AdamW betas=(0.9, 0.95)，矩阵 weight decay=0.1；
- 梯度裁剪阈值 1.0，LR=1e-3，cosine 最低比例 0.1；
- 每 100 步及结束时验证全部 488 条序列、998,936 个预测位置；
- 训练 seed 控制初始化与数据顺序，同一个 seed 内配对比较。

总步数固定，所以延长 warmup 也缩短了之后的 cosine 衰减阶段。本实验测的是
固定训练预算下完整 LR 调度的变化，不能把结果只归因于初期稳定性。

## 结果

| warmup / seed | 最终验证 perplexity | 最后 100 步训练 loss 均值 | 会话耗时，秒 | 更新吞吐中位数，tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 100 / 2026 | 42.2896 | 3.698737 | 1440.03 | 83,813 |
| 100 / 2027 | 41.0086 | 3.684072 | 1448.77 | 84,103 |
| 300 / 2026 | 38.7065 | 3.611185 | 1426.05 | 84,034 |
| 300 / 2027 | 38.6626 | 3.624906 | 1433.06 | 83,725 |

Perplexity 为 `exp(验证 loss)`。训练均值不是对整个训练集重新评估的结果。
吞吐取 step ≥ 10 的中位数，仅含参数更新；会话耗时包含验证与保存，不含启动。
四组峰值 PyTorch allocated 显存均约 8.656 GiB。

| Step | 100 / 2026 验证 loss | 300 / 2026 | 100 / 2027 | 300 / 2027 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.511905 | 10.511905 | 10.506346 | 10.506346 |
| 100 | 6.498775 | 6.751587 | 6.503611 | 6.815671 |
| 300 | 5.559677 | 5.612340 | 5.541026 | 5.574932 |
| 500 | 5.096662 | 5.036577 | 5.048938 | 4.996559 |
| 1000 | 4.440739 | 4.330690 | 4.406564 | 4.325114 |
| 2000 | 3.927013 | 3.843663 | 3.895033 | 3.841125 |
| 3052 | 3.744542 | 3.656008 | 3.713781 | 3.654872 |

warmup=300 前期下降较慢，在上述采样点中第 500 步已反超，并保持到结束。
四组各自的全部验证点持续下降，记录的 loss 和梯度范数均有限。

| warmup / seed | 裁剪次数 / 3052 | 裁剪前范数峰值 | 最后 100 步范数均值 |
| --- | ---: | ---: | ---: |
| 100 / 2026 | 28 | 4.749 | 0.395 |
| 100 / 2027 | 28 | 4.807 | 0.404 |
| 300 / 2026 | 47 | 4.820 | 0.386 |
| 300 / 2027 | 49 | 4.698 | 0.409 |

延长 warmup 并没有减少全程裁剪次数，也没有一致降低梯度峰值。收益证据来自
固定预算下的验证 loss，而非“曲线更平”或“裁剪次数更少”。

## 复现和产物

原 warmup=100 基线分别保存在 `runs/m02_99m_lr1e3/` 和
`runs/m02_99m_lr1e3_seed2027/`。新增结果位于：

- `runs/m02_99m_lr1e3_warmup300_seed2026/`；
- `runs/m02_99m_lr1e3_warmup300_seed2027/`。

各目录包含 checkpoint、summary 和 TensorBoard 日志。以下命令用于说明复现
设置，目录已经存在，重跑需换新目录。seed=2027 时同步替换 seed 与输出目录后缀。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python scripts/train/pretrain.py \
  --model-config configs/model/p099m.yaml \
  --data-dir data/tokenized/m01_fineweb_100m \
  --device cuda:0 --precision bf16 \
  --learning-rate 1e-3 --warmup-steps 300 --min-lr-ratio 0.1 \
  --batch-size 4 --grad-accum-steps 4 --epochs 1 --seed 2026 \
  --beta1 0.9 --beta2 0.95 --weight-decay 0.1 --max-grad-norm 1.0 \
  --eval-every 100 --save-every 100 --eval-batches 0 \
  --output-dir runs/m02_99m_lr1e3_warmup300_seed2026
```

## Checkpoint 加载与生成验收

2026-09-23 对两个 warmup=300 的最终 checkpoint，各检查三个固定 prompt。
六次均严格加载成功，step=3052。使用 CPU / FP32，最多 32 个新增 token，
temperature=0.7、top-k=50、top-p=0.9、生成 seed=2026。以下为原样摘录：

| 训练 seed | Prompt | 新增文本摘录 |
| --- | --- | --- |
| 2026 | The future of artificial intelligence | `and the ability to communicate with others.` |
| 2026 | Water is important because | `the amount of water in the air is high in the air, and the amount of water in the air in the air is low in the air.` |
| 2026 | The solar system consists of | `a 10-foot (50-foot) water tank.` |
| 2027 | The future of artificial intelligence | `In the 19th century, the Soviet Union was the first European country in the world.` |
| 2027 | Water is important because | `it has a lot of potential to provide a safe and efficient environment for the home.` |
| 2027 | The solar system consists of | `a 100% total system of power.` |

两个产物都能加载、生成，但仍存在重复、跑题和明显事实错误。验证 loss 改善
并不代表已有可靠语言能力；不能根据这六个样例对整体生成质量作统计结论。

复现其中一项，替换 checkpoint 的 seed 后缀和 prompt 可检查其余样例：

```bash
PYTHONNOUSERSITE=1 python scripts/inference/pretrained_checkpoint_demo.py \
  --checkpoint runs/m02_99m_lr1e3_warmup300_seed2026/checkpoint.pt \
  --device cpu --precision fp32 \
  --prompt "The future of artificial intelligence" \
  --max-new-tokens 32 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 2026
```

## 下一步：M3 单卡与双卡 DDP 对照

M2 已完成 LR 筛选、配对 seed 复核及 warmup 对比；其他超参数可留作后续研究。
M3 优先固定已选架构、数据和训练参数，验证分布式实现，不同时扩大模型和数据。
当前 `pretrain.py` 只有单卡实现，以下为待实现的实验设计，不是现成双卡命令。

1. 实现双卡 DDP 入口：按 local rank 绑定 GPU，初始化进程组，正确分配数据，
   处理梯度同步、验证指标聚合及 checkpoint/RNG 恢复。只由 rank 0 写共享日志
   和 checkpoint；每个 rank 的随机状态与数据进度应可恢复。
2. 先做 CPU/FP32 或 GPU/FP32 小规模更新对照：相同初始化、相同全局 batch、
   相同 seed、dropout=0，检查 loss 和一次更新后的参数差异。浮点归约顺序不同，
   使用明确误差容限而非强求逐位一致；容限及实际差异写进 M3 报告。
3. 在 99M 上比较单卡与双卡的 100～300 步 BF16 训练。单卡 `4 × 4 = 16`
   条/更新；双卡每卡 `4 × 2`，乘 2 个 rank 仍为 16。全局数据顺序必须相同，
   不能只保证每个 rank 设置相同 seed。短跑沿用完整预算的 LR 计划，明确
   100 步只覆盖 warmup 的一部分。
4. 核对尾部和验证：训练集最后一个全局 batch 为 12 条，双卡每卡 6 条，正确
   加权最后不足 micro-batch 的样本；验证覆盖全部 488 条，不重复计数。以后
   出现不均分尾部时需要另行处理，不能静默丢弃或补重复样本。
5. 做双卡中断恢复对照，再完成约 1 亿 token 的双卡运行。比较相同 token
   预算的验证 loss、全局吞吐、每卡显存及端到端时间，不预设双卡必然加速。

全局吞吐按所有 rank 的总 token 数和同步后的墙钟时间计，不把单卡吞吐直接
乘以 2。当前 99M 单卡可以容纳，DDP 用于学习同步与性能；FSDP 留到需要学习
分片或更大模型的阶段。正式的泛化结论仍需独立评估集，本轮验证集已用于调参。
