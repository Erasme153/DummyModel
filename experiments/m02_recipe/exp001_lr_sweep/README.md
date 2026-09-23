# M2：99M 学习率对比

## 当前结论

已完成 seed=2026 的五组 LR 对比，以及 `6e-4 / 1e-3` 在 seed=2027 的配对复核。
七组均为 `complete`，各完成 3052 次更新、99,999,744 个输入 tokens。

**当前选择峰值 LR=1e-3 作为该模型和预算下的训练基线。** 它在两个训练 seed
下均优于 `6e-4`，平均验证 loss 由 3.779156 降至 3.729161。`5e-3` 明显退化。
本轮 LR 筛选和初步 seed 复核已完成；其他训练参数研究按需开展。

后续更新：固定 LR=1e-3 的 warmup=100/300 对比及两个新 checkpoint 的生成检查
已完成，当前基线采用 warmup=300；详见 [Warmup 报告](../exp002_warmup/README.md)。

结果根据各组 `summary.json` 和 TensorBoard 原始标量核对：最终 step 和验证
loss 一致（允许事件文件浮点存储误差），每组有 3052 条训练和 32 条验证记录。
七组保存的 `contract` 仅 LR 和训练 seed 不同，模型、数据指纹及其他训练设置一致。

## 固定设置

| 项目 | 设置 |
| --- | --- |
| 配置 | [p099m.yaml](../../../configs/model/p099m.yaml) |
| 架构 | 12 层，hidden=768，MLP=2048，Q heads=12，KV heads=3，共享 embedding |
| 参数量 | 98,913,024 |
| 数据 | `data/tokenized/m01_fineweb_100m/`，FineWeb-Edu，复用 M1 数据 |
| Tokenizer | Mistral 32K，正文不加 BOS，文档末尾加 EOS |
| 训练预算 | 1 epoch，99,999,744 输入 tokens，99,950,916 有效预测位置 |
| 序列长度 / 有效 batch | 2048 / 16 条；micro-batch=4，梯度累积=4 |
| 验证 | 全部 488 条序列，998,936 个预测位置；初始、每 100 步及结束时 |
| 精度 | CUDA BF16 autocast，参数和 AdamW 状态为 FP32 |
| AdamW | betas=(0.9, 0.95)，矩阵 weight decay=0.1，RMSNorm 不衰减 |
| 梯度裁剪 | 累积后做全局 L2 norm clipping，阈值 1.0 |
| LR 调度 | 100 步线性 warmup，cosine 衰减至各自峰值的 10% |
| 训练 seed | 初筛 2026；配对复核 2027，同时改变初始化与训练数据排列 |

数据来源、划分和精确去重的限制见 [M1 报告](../../m01_pretraining/exp001_39m/README.md)。
所有比较使用同一验证集；这里不构成等计算预算的 scaling 实验。

## 完整训练结果

### Seed=2026：五组学习率

| 峰值 LR | 最终验证 loss ↓ | 验证 perplexity ↓ | 最后 100 步训练 loss 均值 | 会话秒数 |
| --- | ---: | ---: | ---: | ---: |
| 1e-4 | 4.559373 | 95.5235 | 4.535344 | 1437.99 |
| 3e-4 | 3.951896 | 52.0339 | 3.915144 | 1432.94 |
| 6e-4 | 3.792916 | 44.3856 | 3.746580 | 1433.94 |
| **1e-3** | **3.744542** | **42.2896** | **3.698737** | 1440.03 |
| 5e-3 | 5.536487 | 253.7849 | 5.520574 | 1452.29 |

Perplexity 为 `exp(验证 loss)`。训练均值来自更新前各 batch，不是整个训练集
的固定模型评估值。会话时间包括验证、保存，不含启动加载；五组均从 step 0
开始，`1e-3` 此次实际为完整会话，并非按旧计划在 step 100 暂停后恢复。

| Step | 1e-4 | 3e-4 | 6e-4 | 1e-3 | 5e-3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10.511905 | 10.511905 | 10.511905 | 10.511905 | 10.511905 |
| 100 | 7.228041 | 6.791522 | 6.566803 | 6.498775 | 6.515633 |
| 500 | 5.712459 | 5.235749 | 5.122209 | 5.096662 | 6.025765 |
| 1000 | 5.131691 | 4.652737 | 4.487828 | 4.440739 | 6.079302 |
| 2000 | 4.692151 | 4.120696 | 3.965521 | 3.927013 | 5.821403 |
| 3052 | 4.559373 | 3.951896 | 3.792916 | 3.744542 | 5.536487 |

`5e-3` 在 step 100 看起来接近 `1e-3`，之后明显落后：验证 loss 在 step
400、600、700、800、1100 出现反弹。这说明短跑仅能筛查明显问题，不能替代
完整预算对比。它最终完成且 loss/梯度均有限，但收敛质量明显退化；不标为 NaN 发散。

### 配对 seed 复核

| 训练 seed | 6e-4 验证 loss | 1e-3 验证 loss | 1e-3 改善量 |
| --- | ---: | ---: | ---: |
| 2026 | 3.792916 | 3.744542 | 0.048374 |
| 2027 | 3.765396 | 3.713781 | 0.051616 |
| 均值 | 3.779156 | **3.729161** | **0.049995** |

`6e-4` 两次结果范围为 [3.765396, 3.792916]；`1e-3` 为
[3.713781, 3.744542]。两次排名一致，支持选择 `1e-3`；两个 seed 仅用于初步
稳健性检查，不作统计显著性或全局最优声明。`1e-3` 与 `5e-3` 之间仍有未测试候选。

Seed=2027 两组的初始验证 loss 均为 10.506346，后续验证点均下降。
`6e-4 / 1e-3` 最后 100 步训练 loss 均值分别为 3.737003 / 3.684072，
会话耗时分别为 1547.37 / 1448.77 秒。不同 seed 的数据顺序不同，不用最后一个
batch 的 loss 比较模型优劣。

### 梯度与性能

| LR / seed | 裁剪次数 / 3052 | 裁剪比例 | 最后 100 步裁剪前范数均值 | 更新吞吐中位数，tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1e-4 / 2026 | 2783 | 91.19% | 1.192 | 83,359 |
| 3e-4 / 2026 | 58 | 1.90% | 0.667 | 83,711 |
| 6e-4 / 2026 | 31 | 1.02% | 0.479 | 83,494 |
| 1e-3 / 2026 | 28 | 0.92% | 0.395 | 83,813 |
| 5e-3 / 2026 | 2559 | 83.85% | 2.919 | 84,051 |
| 6e-4 / 2027 | 31 | 1.02% | 0.487 | 83,779 |
| 1e-3 / 2027 | 28 | 0.92% | 0.404 | 84,103 |

吞吐中位数取 step ≥ 10，只计训练更新；七组 PyTorch 峰值 allocated 显存
均约 8.656 GiB，不是整张 GPU 占用。

`train/gradient_norm` 是裁剪前的范数。记录 1.2 时，梯度按约 `1/1.2`
缩放后再交给 AdamW；日志大于 1 不代表裁剪失效。

`1e-4` 后期梯度稳定在略高于 1 的范围，伴随相对缓慢的收敛。`5e-3`
则有明显尖峰：step 1759 的裁剪前范数达到 **59.592865**，并伴随较差的验证
曲线。两者均频繁裁剪，但不能据此认定原因相同，也不能把裁剪比例直接作为质量排名。
不同 LR 会改变参数轨迹；AdamW 的真实更新幅度不等于 `LR × 原始梯度范数`。

## 复现与产物

在项目根目录、已激活项目环境后执行以下模板。已存在的输出目录不能覆盖，
复跑需改新目录；每个 LR/seed 组合从头开始，不跨实验加载 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python scripts/train/pretrain.py \
  --model-config configs/model/p099m.yaml \
  --data-dir data/tokenized/m01_fineweb_100m \
  --device cuda:0 --precision bf16 --seed 2026 --epochs 1 \
  --batch-size 4 --grad-accum-steps 4 \
  --warmup-steps 100 --min-lr-ratio 0.1 \
  --beta1 0.9 --beta2 0.95 --weight-decay 0.1 --max-grad-norm 1.0 \
  --eval-every 100 --save-every 100 --eval-batches 0 \
  --learning-rate 1e-3 --output-dir runs/m02_99m_lr1e3
```

| LR | seed | 已有产物目录 |
| --- | ---: | --- |
| 1e-4 | 2026 | `runs/m02_99m_lr1e4/` |
| 3e-4 | 2026 | `runs/m02_99m_lr3e4/` |
| 6e-4 | 2026 | `runs/m02_99m_lr6e4/` |
| 1e-3 | 2026 | `runs/m02_99m_lr1e3/` |
| 5e-3 | 2026 | `runs/m02_99m_lr5e3/` |
| 6e-4 | 2027 | `runs/m02_99m_lr6e4_seed2027/` |
| 1e-3 | 2027 | `runs/m02_99m_lr1e3_seed2027/` |

各目录保留 `checkpoint.pt`、`summary.json` 和 `tensorboard/`。
`runs/m02_99m_smoke/` 是 step 100 恢复到 110 的独立短跑，验证 loss
6.670861，状态 `paused`，不纳入排名。

## 已有 checkpoint 生成检查

此前对 seed=2026 的前三组进行了 CPU / FP32 严格加载，均返回
`<All keys matched successfully>`。固定 prompt 为
`The future of artificial intelligence`，生成长度 32，temperature=0.7、
top-k=50、top-p=0.9、生成 seed=2026，摘录如下：

| LR | 新增文本摘录 |
| --- | --- |
| 1e-4 | `The first century of the world is the largest of the world, the world’s world.` |
| 3e-4 | `The National Science Foundation (NIH) is a professor of the National Science Foundation (BBS)` |
| 6e-4 | `is a new, more complex and comprehensive way to understand and to understand the different forms of communication and communication` |

仍有重复、实体错误和空泛表达。新增四组本次核对的是训练摘要和事件日志，
尚未追加统一生成检查；不能由验证 loss 的改善直接断言生成能力已合格。

## 后续工作（已更新）

1. warmup 对比已完成：两个 seed 均支持 300 步，所选新 checkpoint 的三个固定
   prompt 生成检查已完成，结果见 [Warmup 报告](../exp002_warmup/README.md)。
2. 下一步可进入 M3 单卡/双卡 DDP 一致性与性能实验，固定 LR=1e-3、warmup=300。
3. 若研究 scaling，另设模型规模与计算预算对照，不把本轮 LR 当作其他规模的
   最优值。额外 seed=2028 或 LR=2e-3 的细化为可选项。

生成检查命令：

```bash
PYTHONNOUSERSITE=1 python scripts/inference/pretrained_checkpoint_demo.py \
  --checkpoint runs/m02_99m_lr1e3/checkpoint.pt \
  --device cpu --precision fp32 \
  --prompt "The future of artificial intelligence" \
  --max-new-tokens 32 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 2026
```

已完成的 warmup 对照命令（仅供复现，重跑需换新输出目录）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python scripts/train/pretrain.py \
  --model-config configs/model/p099m.yaml \
  --data-dir data/tokenized/m01_fineweb_100m \
  --device cuda:0 --learning-rate 1e-3 --warmup-steps 300 --seed 2026 \
  --output-dir runs/m02_99m_lr1e3_warmup300_seed2026
```

本轮结论限定于当前 99M 模型、数据与约 1 亿 token 预算。验证集已用于超参数选择，
最低验证 loss 不是独立测试集成绩。后续新变量实验单独建一份 README，保留本轮
七组结果；无需新增 manifest 或训练配置管理文件。
