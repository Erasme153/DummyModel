# M7：Top-k MoE、QB 路由与双卡 EP

## 结论

M7 已完成。与 active parameters 约相同的 99M dense 基线相比，四专家、
每 token 选两专家的 MoE 在两个 seed 上分别降低独立验证 loss **0.023250**
和 **0.019524**；双卡 DDP 的训练更新吞吐低约 **28.2%**。辅助负载均衡
使专家使用更均匀；容量限制会丢路由，本次没有改善 loss。QB 在两个 seed
上使路由更均匀，但平均 loss 与辅助损失版 top-k 几乎相同，吞吐还低约
**6.9%**。双卡 EP 每卡只持有两个专家，训练质量与完整专家 DDP 接近；
稳态更新吞吐低约 **8.3%**，已测最高每卡显存仅降低 **0.315 GiB**。
现有对照足以验收 M7，不继续追加 MoE 参数搜索。

## 设置与结果

共用 M1 FineWeb-Edu 训练集的 **99,999,744 输入 tokens**、2048 长度、
Mistral 32K tokenizer；同集验证为 488 条序列，独立验证统一使用 M4 的
4,882 条序列、9,993,454 个预测位置。BF16、AdamW、LR=1e-3、
warmup=300、cosine 最低 LR 比例 0.1、全局 batch=16、梯度裁剪 1.0、
1 epoch / **3052** 次更新。除表中注明的单卡 dense seed 2027 外，
均使用两张 H20、每卡 batch=4、累积 2 次。

MoE 配置为 `moe_p099m_e4k2.yaml`：12 层、4 个 SwiGLU 专家、top-2，
总参数 **155,572,992**，每 token 激活约 **98,949,888** 个参数；
`p099m.yaml` dense 基线为 **98,913,024** 个参数。两者 active parameter
数接近，实际计算量还受路由和分发操作影响。QB 使用分位数偏置，
不加辅助负载损失；EP 使用无容量限制的 top-k + 0.01 辅助损失。

| `runs/` 目录 | 路由 / 设置 | seed | 同集 loss ↓ | 独立 loss ↓ |
| --- | --- | ---: | ---: | ---: |
| `m03_99m_ddp` | dense，双卡 | 2026 | 3.657155 | 3.627178 |
| `m02_99m_lr1e3_warmup300_seed2027` | dense，单卡 | 2027 | 3.654872 | 3.624344 |
| `m07_moe_e4k2_aux0_nocap` | top-k，aux=0 | 2026 | 3.639671 | 3.610085 |
| `m07_moe_e4k2_aux0p01_nocap` | top-k，aux=0.01 | 2026 | **3.634160** | **3.603928** |
| `m07_moe_e4k2_aux0p01_cap1p25` | top-k，aux=0.01，capacity=1.25 | 2026 | 3.640213 | 3.610381 |
| `m07_moe_e4k2_aux0p01_nocap_seed2027` | top-k，aux=0.01 | 2027 | 3.635582 | 3.604820 |
| `m07_moe_e4k2_qb_seed2026` | QB，aux=0 | 2026 | 3.639286 | 3.608427 |
| `m07_moe_e4k2_qb_seed2027` | QB，aux=0 | 2027 | **3.629576** | **3.600704** |
| `m07_moe_ep_topk_seed2026` | top-k，aux=0.01，双卡 EP | 2026 | **3.631202** | **3.602258** |

所有组状态为 `complete`，各见 99,999,744 个输入 tokens。两 seed 的
top-k 独立 loss 均值为 **3.604374**，QB 为 **3.604566**；QB 的
2026 结果稍差、2027 稍好，不能据此宣称质量优势。MoE 相对 dense 的
loss 收益也只在此模型、数据和预算下成立。

无辅助损失的 top-k 到末步仍有一层 **99.25%** 的专家任务落在同一
rank 对应的两个专家上，路由辅助指标为 **1.527**；aux=0.01 的双卡
DDP 末步最大 rank 份额为 **65.42%**，辅助指标为 **1.038**。
前者的指标仍被记录，但乘以零后不参与训练目标。
capacity=1.25 的丢弃路由比例从首个记录点 **13.08%** 降到末步
**0.75%**，末步未出现整 token 无专家可用；其独立 loss 比无容量限制组
高 **0.006453**。QB 两个 seed 末步各专家选择比例分别约为
**19.85%–27.83%**、**22.12%–26.36%**，明显更均匀，且不丢路由。
两个 QB checkpoint 的偏置绝对值约为 138–357，但同一层四专家的
最大偏置差分别仅 0.76、1.10：共同平移不改变选择，不能用跨层
绝对值判断发散。

## EP 性能与 profiling

EP 将专家 0、1 放在 rank 0，专家 2、3 放在 rank 1；非专家权重同步，
token 通过双向 all-to-all 到所属专家并按原位置加权合并。每卡驻留
**98,949,888** 个模型参数，比完整专家 DDP 少 **36.4%**。
训练前的 CPU/Gloo 测试覆盖前向、梯度、一次更新、零 token 专家和断点
恢复；两卡 H20/BF16 完整预算及分片 checkpoint 独立评估也已通过。

下表吞吐由 TensorBoard 第 **301–3000** 步的 `train/tokens_per_second`
反推总更新时间，再以总输入 tokens 除以该时间；不含验证与 checkpoint。
峰值取所有训练会话记录的最大 `torch.cuda.max_memory_allocated`，
不是两卡求和或 `nvidia-smi` 占用。均为 seed 2026。

| 双卡训练 | 全局 tokens/s ↑ | 每卡最高 allocated，GiB ↓ |
| --- | ---: | ---: |
| dense DDP | 163,896 | 9.032 |
| top-k DDP，aux=0.01，无容量 | 117,708 | 12.665 |
| QB DDP | 109,581 | 12.622 |
| top-k EP，aux=0.01，无容量 | 107,889 | 12.351 |

EP 与 top-k DDP 的 step=0 验证 loss 只差约 0.000001；配对 40 步
短跑的最终验证 loss 为 **7.881047 / 7.880728**（EP / DDP）。
完整训练和独立验证分别只差 **0.002958 / 0.001671**，没有持续质量
退化，也不能把这点差异当作 EP 质量增益。EP 最后一步第 0 层有
**69.55%** 的专家任务落在 rank 1，第 11 层为 **65.35%**；
负载失衡仍会限制并行效率。早期第 40 步部分层几乎完全落到一张卡，
DDP 对照也有同类路由偏斜，不是 EP 分发错误。

40 步 Nsight 轨迹位于 `/tmp/m07_moe_ddp_nsys.nsys-rep` 与
`/tmp/m07_moe_ep_nsys.nsys-rep`，EP 中可见大量 NCCL `SendRecv`。
但启用 `--pytorch=autograd-nvtx` 后，第 20–39 步记录的吞吐变为
DDP **64,343**、EP **88,282** tokens/s；未剖析的对应窗口为
DDP **117,915**、EP **97,793**。剖析显著扰动两条路径，不能用
其速度排序代替正常训练结论。Nsight 整段 kernel 汇总还混合初始化、
验证和保存，不能把汇总的 NCCL 时间直接当作纯训练通信成本。
此外，DDP 恢复段主要每 500 步验证、每 1000 步保存，EP 恢复段
每 100 步均验证和保存；两份 `session_elapsed_seconds` 不可直接比较。

## 复现与产物

在项目根目录执行；已有 `runs/` 目录不能覆盖，复现时更换输出目录。
下列 DDP 命令复现无容量 top-k 主组。其余 DDP 组仅改一项：
aux=0 使用 `--moe-aux-loss-coef 0`；容量组增加
`--moe-capacity-factor 1.25`；QB 使用
`--moe-routing qb --moe-aux-loss-coef 0`；第二个 seed 使用 `--seed 2027`。
已完成的 dense 对照直接复用，不重新训练。

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py --data-dir data/tokenized/m01_fineweb_100m \
  --model-config configs/model/moe_p099m_e4k2.yaml \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 \
  --learning-rate 1e-3 --warmup-steps 300 --moe-aux-loss-coef 0.01 \
  --seed 2026 --eval-every 500 --save-every 1000 --log-every 100 \
  --output-dir runs/m07_moe_e4k2_aux0p01_nocap

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ep.py --data-dir data/tokenized/m01_fineweb_100m \
  --model-config configs/model/moe_p099m_e4k2.yaml \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 \
  --learning-rate 1e-3 --warmup-steps 300 --moe-aux-loss-coef 0.01 \
  --seed 2026 --eval-every 100 --save-every 100 --log-every 10 \
  --output-dir runs/m07_moe_ep_topk_seed2026 --stop-after-steps 100
```

EP 第 100 步正常后，保留同样训练参数、删除 `--stop-after-steps` 和
`--output-dir`，改为 `--resume runs/m07_moe_ep_topk_seed2026/checkpoint.pt`
续至 3052 步。Nsight 使用两次独立的 40 步短跑：

```bash
PROFILE_ARGS=(--data-dir data/tokenized/m01_fineweb_100m
  --model-config configs/model/moe_p099m_e4k2.yaml --device cuda --precision bf16
  --batch-size 4 --grad-accum-steps 2 --learning-rate 1e-3 --warmup-steps 300
  --moe-aux-loss-coef 0.01 --seed 2026 --eval-batches 4
  --eval-every 1000 --save-every 1000 --log-every 1 --stop-after-steps 40)
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 nsys profile \
  --trace=cuda,nvtx --pytorch=autograd-nvtx --sample=none --cpuctxsw=none \
  -o /tmp/m07_moe_ddp_nsys torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ddp.py "${PROFILE_ARGS[@]}" --output-dir runs/m07_profile_moe_ddp
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 nsys profile \
  --trace=cuda,nvtx --pytorch=autograd-nvtx --sample=none --cpuctxsw=none \
  -o /tmp/m07_moe_ep_nsys torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_ep.py "${PROFILE_ARGS[@]}" --output-dir runs/m07_profile_moe_ep
```

独立评估示例：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python scripts/eval/base_eval.py loss \
  --checkpoint runs/m07_moe_ep_topk_seed2026/checkpoint.pt \
  --data-dir data/tokenized/m04_fineweb_1b --device cuda:0 --precision bf16 \
  --batch-size 4 --output runs/m07_eval/val_m07_moe_ep_topk_seed2026.json
```

各训练的 checkpoint、summary 和 TensorBoard 日志均在对应的 `runs/`
目录；独立验证 JSON 在 `runs/m07_eval/`。EP 的 `checkpoint.pt` 是
元数据，另有两个 rank 的分片文件；独立评估入口会合并分片并检查共享权重。

## 后续

按项目路线进入 M8。当前训练入口的 `--resume` 要求原数据与训练约定
一致，不能直接把 M4 完成的 base checkpoint 当成新阶段续训；先实现
独立的 midtraining 初始化入口，并准备与 M4 训练集不重叠的新数据，
再从同一 base checkpoint 做固定预算的 continuation/cooldown 对照。
