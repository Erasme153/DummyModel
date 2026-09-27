# M3：单卡 / 双卡 DDP 一致性与性能

## 结论

本实验已完成。固定全局 batch=16、seed=2026 和 1 epoch 预算，单卡与双卡
DDP 均完成 3052 次更新、99,999,744 个输入 tokens。最终全量验证 loss 分别为
**3.657353** 和 **3.657155**，相差 0.000198；两组的 32 个验证点均持续下降。
双卡在相同区间的全局训练吞吐为单卡的约 **1.95 倍**，每卡 PyTorch allocated
峰值显存高约 0.38 GiB。100→400→3052 的恢复和最终 checkpoint 严格加载均
通过。40 步 Nsight 短程剖析也完成，稳定区间的更新耗时比约 1.94×；FSDP2
和正式能力评测未开展。

## 目的与固定参数

只改变并行方式，不重新筛选 LR。沿用 M2 的 `p099m.yaml`、现有
`data/tokenized/m01_fineweb_100m`、seed=2026、LR=1e-3、warmup=300、
BF16、AdamW、梯度裁剪阈值 1.0、1 epoch。完整计划仍为 3052 次更新。
所有命令在项目根目录、已经安装项目依赖的环境内执行。

| 对照 | 进程 / GPU 数 | 每卡 micro-batch | 每卡累积次数 | 全局 batch |
| --- | ---: | ---: | ---: | ---: |
| 单卡 | 1 | 4 | 4 | 16 |
| 双卡 DDP | 2 | 4 | 2 | 16 |

两组从相同 seed **重新初始化**，不加载 M2 最终模型。保持全局 batch、数据顺序、
更新次数和 LR 计划一致；卡数翻倍不代表 LR 要翻倍。先确认两张卡空闲，再开展性能比较，
两组不要同时跑。

## 操作顺序

### 1. 单卡对照：先跑 100 步

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
python scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 4 \
  --output-dir runs/m03_99m_single \
  --stop-after-steps 100
```

### 2. 双卡对照：先跑 100 步

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 \
torchrun --standalone --nproc-per-node=2 scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 2 \
  --output-dir runs/m03_99m_ddp \
  --stop-after-steps 100
```

多卡使用 `--device cuda`，脚本用 `LOCAL_RANK` 绑定可见 GPU，不传 `cuda:0` 或
`cuda:1` 把所有进程指定到同一张卡。启动日志应显示两个 rank 分别使用 cuda:0/1，
主日志显示 `world_size=2 global_batch=16`。已有输出目录不会被覆盖。

### 3. 检查曲线，再恢复至 400 步

```bash
tensorboard --logdir runs
```

比较 step=0、100 的验证 loss、前 100 步的训练 loss/gradient_norm，确认
LR、tokens_seen 完全对齐且没有 NaN/Inf、卡住或样本重复计数。浮点求和顺序变化，
BF16 下不要求两条曲线逐位相同；若出现明显、持续的分叉，先用相同小模型和 FP32
定位，不能仅靠放宽容差通过。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
python scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 4 \
  --resume runs/m03_99m_single/checkpoint.pt \
  --stop-after-steps 400

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 \
torchrun --standalone --nproc-per-node=2 scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 2 \
  --resume runs/m03_99m_ddp/checkpoint.pt \
  --stop-after-steps 400
```

这里只延后暂停点，余弦 LR 仍按完整 3052 步计划运行。恢复时卡数、每卡 batch、
累积次数及其他训练约定必须保持原样。新脚本不直接恢复旧 `pretrain.py` 的 v1
checkpoint，也不支持单卡 checkpoint 切成双卡续训；这是本次实验有意保留的边界。

### 4. 从 400 步继续完成完整预算

两组第 400 步的 checkpoint 分别恢复至 3052 步；不传暂停点，沿用 checkpoint
中的模型、数据、batch、LR、精度及 seed。结果目录已存在，不能作为新训练的
输出目录再次使用。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
python scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 4 \
  --resume runs/m03_99m_single/checkpoint.pt

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 \
torchrun --standalone --nproc-per-node=2 scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 2 \
  --resume runs/m03_99m_ddp/checkpoint.pt
```

### 5. 实测结果

核对两组 `summary.json`：模型均为 98,913,024 参数，模型配置、tokenizer 和
训练/验证数据指纹一致；contract 仅在 GPU 数和每卡累积次数上不同。两组最终
状态均为 `complete`，输入 tokens 均为 99,999,744，有效预测位置数均为
99,950,916。TensorBoard 各有连续 3052 条训练记录和 32 个全量验证点；每步
`train/learning_rate` 与全局 `train/tokens_seen` 完全一致，记录的训练 loss 与
梯度范数均为有限数。

| Step | 单卡验证 loss | 双卡验证 loss | 双卡 − 单卡 |
| ---: | ---: | ---: | ---: |
| 0 | 10.511905 | 10.511905 | 0.000000 |
| 100 | 6.751990 | 6.752335 | +0.000345 |
| 200 | 6.009499 | 6.014434 | +0.004935 |
| 400 | 5.261559 | 5.264299 | +0.002740 |
| 1000 | 4.335618 | 4.331192 | -0.004426 |
| 2000 | 3.844881 | 3.844933 | +0.000052 |
| 3000 | 3.661053 | 3.660404 | -0.000649 |
| 3052 | **3.657353** | **3.657155** | **-0.000198** |

32 个验证点的最大绝对差为 0.006490（step 1700）。最终验证 perplexity 分别为
38.7586 和 38.7509。训练 loss 的最大单步差为 0.035266（step 1140），裁剪前
梯度范数大于 1.0 的更新分别为 48 和 44 次。BF16 算子与跨卡归约的求和顺序
不同，两条轨迹无需逐位相同；当前没有持续扩大的验证差距，单个训练 seed 也
不足以证明两种并行方式在总体质量上完全等价。

| 更新区间 | 单卡全局 tokens/s | 双卡全局 tokens/s | 加速比 | 双卡效率 |
| --- | ---: | ---: | ---: | ---: |
| 102–400 | 84,394 | 164,313 | 1.947× | 97.35% |
| 402–3051 | 84,039 | 164,014 | 1.952× | 97.58% |

400→3052 步恢复会话的耗时：单卡 1242.81 秒，双卡 686.40 秒，相同区间的
会话加速比为 1.811×。会话时间包含验证与 checkpoint 保存，不含此前会话以及
启动时的数据和模型构建；不能与只计训练更新的吞吐混为一谈。
按逐步吞吐反推，该会话内的训练更新共耗时约 1034.34 / 530.22 秒，余下约
208.47 / 156.18 秒包含验证、写盘及其他循环开销；现有指标不能继续精确拆分。

| PyTorch allocated 峰值，GiB | 单卡 | 双卡 rank 0 | 双卡 rank 1 |
| --- | ---: | ---: | ---: |
| 各会话记录的最大值 | 8.656 | 9.032 | 9.022 |

两组均经过 100→400→3052 的 checkpoint 恢复。最终两个 checkpoint 使用
`pretrained_checkpoint_demo.py` 在 CPU/FP32 下严格加载，均返回
`<All keys matched successfully>`，报告 step=3052、tokens_seen=99,999,744；
固定 prompt `The future of artificial intelligence` 贪心生成 1 个 token，
两者均输出 `,`。这只验证加载与一次生成，不作为语言能力评测。每组产物仍只有
`checkpoint.pt`、`summary.json` 和 `tensorboard/`，保存在对应的本地 `runs/` 目录。

`train/tokens_per_second` 是**全局输入 tokens/s**，不用再乘卡数；包含训练更新、
梯度通信和更新前 barrier，不含验证、checkpoint 写盘和后续指标汇总。
比较相同区间，去掉恢复后的首次更新；第二个区间还去掉只有 12 条序列的
epoch 尾步 3052。聚合吞吐用 `总 tokens / 总更新耗时`，不直接平均逐步速度。
速度比为双卡吞吐 / 单卡吞吐，并行效率为速度比 / 2。
`session_elapsed_seconds` 包含本次运行的验证与保存，但不含此前的数据/模型初始化。

每次启动和恢复都会重置峰值计数，上表取各会话记录的最大值。
`train/peak_memory_gib_rank0/1` 分别记录各卡在本次进程中的累计峰值，
`train/peak_memory_gib` 取最大值，不是两卡之和，也不是 nvidia-smi 的全部显存占用。
DDP 每卡保留完整模型、梯度和 AdamW 状态，另有同步开销；不能仅凭峰值把
增加的约 0.38 GiB 精确归因到某一种缓冲。

## 短程 Nsight profiling

从相同 seed 分别新训 40 步，维持上表的每卡 batch、累积次数和全局 batch=16；
`--stop-after-steps 40` 不改变原定 3052 步 LR 计划。两组均在 step=0、40 做
全量验证和 checkpoint 保存，状态为 `paused`，各处理 1,310,720 个输入 tokens。
最终验证 loss 为 7.831694 / 7.831699。训练日志在
`runs/m03_profile_single/`、`runs/m03_profile_ddp/`；Nsight 原始报告在
`tmp/m03_single_nsys.nsys-rep`、`tmp/m03_ddp_nsys.nsys-rep`。这两次是新的
短跑，不能与上述 3052 步 checkpoint 混用。以下为复现参数；现有输出目录
已占用，重跑须更换训练与 Nsight 输出路径。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
nsys profile --trace=cuda,nvtx --pytorch=autograd-nvtx \
  --sample=none --cpuctxsw=none -o /tmp/m03_single_nsys \
python scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 4 \
  --output-dir runs/m03_profile_single --stop-after-steps 40

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 \
nsys profile --trace=cuda,nvtx --pytorch=autograd-nvtx \
  --sample=none --cpuctxsw=none -o /tmp/m03_ddp_nsys \
torchrun --standalone --nproc-per-node=2 scripts/train/pretrain_ddp.py \
  --device cuda --batch-size 4 --grad-accum-steps 2 \
  --output-dir runs/m03_profile_ddp --stop-after-steps 40
```

| 第 20–39 步 | 单卡 | 双卡；分 rank 时按 0 / 1 顺序 |
| --- | ---: | ---: |
| 20 次更新耗时，训练日志 | 7.933 秒 | 4.082 秒 |
| 相应 Nsight 时间窗 | 7.947 秒 | 两个 rank 均为 4.105 秒 |
| GPU 有 CUDA kernel 活动的时间占比 | 92.9% | 91.2% / 92.0% |
| NCCL kernel 活动，去重后的时长 | — | 0.041 / 0.098 秒 |
| 其中与其他 kernel 重叠 | — | 0.028 / 0.061 秒 |

训练日志的 20 步全局吞吐为 82,608 / 160,547 tokens/s，双卡加速比 1.944×。
Nsight 时间窗按每个 rank 的第 19 次 AdamW 更新结束至第 39 次结束截取，
包含近似的第 20–39 步；GPU 活动比例是 kernel 时间区间并集 / 窗口时长，
**不是 SM 利用率**。NCCL 时长也不是可直接从训练耗时扣除的纯通信开销：
内核可能等待另一 rank，且部分执行与计算重叠。正常训练吞吐仍以上面的
3052 步实验为准。

整段双卡轨迹中，rank 1 的 NCCL Broadcast 内核累计约 4.59 秒，其中约
1.23 秒位于首步前、约 3.35 秒位于第 40 步后。结合代码中 rank 0 独自写入
checkpoint、随后 `broadcast_object_list` 同步保存结果的流程，这主要是
rank 1 等待写盘，不能算成每步梯度通信耗时。`cuda_gpu_kern_sum` 和
`cuda_api_sum` 默认混合初始化、验证、训练及保存；后者的
`cudaStreamSynchronize` 时间是 CPU 等待时间，不代表 GPU 空闲时间。
本次 PyTorch NVTX 操作名含独立 `op_id`，直接运行 `nvtx_sum` 会产生大量
逐操作行；定位瓶颈应先看时间线的稳定更新区间。

40 步剖析会话总耗时为 28.78 / 17.31 秒，含两次全量验证和保存；
剖析本身也会增加开销，不能用这个会话比值替代正式训练的性能结论。

## 实现与验证

[`pretrain_ddp.py`](../../../scripts/train/pretrain_ddp.py) 对同一全局随机排列按 rank
划分数据，不补重复样本。非最后一个 micro-batch 通过 `no_sync()` 累积梯度，
最后一次同步后才裁剪和更新；尾部按真实样本数加权。验证集无重复划分后汇总，
只有 rank 0 写日志和权重，checkpoint 保存各 rank 的 RNG。当前 99M 的 dropout
为 0；不同卡数的 BF16 归约顺序仍可使参数轨迹略有差异。

Python 3.10 / PyTorch 2.5.1+cu124：此前 `tests/unit` 共 30 项测试通过，包含
4 项 DDP CPU/Gloo 小模型测试，覆盖同步、非均匀尾部、逐 rank RNG 恢复、旧
checkpoint 拒绝及真实 `torchrun` 入口；本轮补足 99M 双 GPU/NCCL/BF16 实测。

```bash
PYTHONNOUSERSITE=1 python -m pytest tests/unit/training/test_pretrain_ddp.py -q
```

完整预算的 DDP 对照与 Nsight 短程剖析均已完成。下一步若要完成 M3 的
分片训练目标，应先实现 FSDP2，再在相同模型、数据、全局 batch 和训练预算下
比较正确性、每卡显存、稳态吞吐及保存/恢复成本；当前还没有 FSDP2 训练入口。

参考：[PyTorch DDP 与 no_sync](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、
[torchrun 与 LOCAL_RANK](https://docs.pytorch.org/docs/stable/elastic/run.html)、
[Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/)。
