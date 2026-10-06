# M6：99M 模型上的 Muon、MuonH 与 AdamH

## 结论

固定 99M 模型、100M 训练 tokens 和 AdamW 辅助参数组后，已完成运行中
MuonW（隐藏层 LR=0.01）的独立验证 loss **3.405980**，低于 AdamW
基线的 **3.627178**，差 **0.221198**；代价是稳态吞吐约低 **15.2%**。
MuonH 的最佳 loss **3.414223**，比 MuonW 高 0.008243；AdamH 的最佳
loss **3.605910**，比 AdamW 低 0.021268。两种 Hyperball 方法在已测
LR 中均以 0.01 最好，但本实验只有一个 seed，后两个小差值不宜解释为
确定的算法优势。M6 的既定比较已完成，下一阶段按项目路线进入 M7 Mini-MoE。

## 设置与结果

所有组使用 `configs/model/p099m.yaml`（98,913,024 参数）、M1 的
FineWeb 100M-token 训练集、同一 tokenizer、序列长度 2048、seed=2026、
双卡 BF16 DDP、每卡 batch=4、梯度累积 2、全局 batch=16。均按完整
3052 步计划运行，warmup=300，cosine 最低 LR 比例 0.1，梯度裁剪阈值
1.0。基线沿用 `runs/m03_99m_ddp`，不重复训练。MuonW、MuonH、AdamH
仅对 Transformer 隐藏层二维矩阵使用对应更新；embedding、输出层和
norm 保持 AdamW，LR=1e-3。Muon 动量 0.95、Newton–Schulz 迭代 5 次。
Hyperball 对每个受约束矩阵固定初始 Frobenius 范数，逐矩阵归一化更新并
投影回该半径；其 LR 是相对更新长度，不与普通学习率直接等价。

表中内部验证使用 M1 约 1M-token 验证集；独立验证统一使用 M4 的
4,882 条序列、9,993,454 个预测位置、BF16。两个 loss 口径不能互换。
吞吐为 TensorBoard `train/tokens_per_second` 在第 301–3000 步、剔除
每 100 步范数记录点后的中位数，单位为全局 tokens/s。

| 运行目录（位于 `runs/`） | 隐藏层 LR | 完成步数 | 内部 loss ↓ | 独立 loss ↓ | 吞吐 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m03_99m_ddp`（AdamW） | 1e-3 | 3052 | 3.657155 | 3.627178 | 164,022 |
| `m06_99m_muon_lr0p01`（MuonW） | 0.01 | 3052 | 3.436665 | **3.405980** | 139,064 |
| `m06_99m_muonh_lr0p005` | 0.005 | 3052 | 3.526596 | 3.493855 | 136,604 |
| `m06_99m_muonh_lr0p01` | 0.01 | 3052 | 3.447049 | **3.414223** | 136,590 |
| `m06_99m_muonh_lr0p02` | 0.02 | 3052 | 3.481835 | 3.450327 | 136,555 |
| `m06_99m_adamh_lr0p005` | 0.005 | 3052 | 3.661682 | 3.631712 | 158,736 |
| `m06_99m_adamh_lr0p01` | 0.01 | 3052 | 3.636615 | **3.605910** | 158,270 |
| `m06_99m_adamh_lr0p02` | 0.02 | 3052 | 3.719334 | 3.688864 | 158,850 |

另有 `m06_99m_muon_lr0p02` 在第 1000 步暂停，当时内部 loss
**4.153995**，同一步 MuonW 0.01 为 **4.042540**。该组没有完整预算
或独立验证结果，不进入终点排名。所有完整组实际各见
99,999,744 个输入 tokens。

Hyperball 两组的受约束参数范数在记录点均保持 **172.419556**；
MuonH/AdamH 在 LR=0.01 时，受约束组最大实际
update/parameter ratio 分别为 **0.009922/0.009956**，末步约为
**0.001**。这是投影约束按预期工作的证据；全模型参数范数仍会因辅助
AdamW 参数变化。无约束 MuonW 0.01 的隐藏层参数范数则从约 170 增至
260。记录的是实际更新范数，不能把 Hyperball 的相对 LR 直接当成
MuonW 的 LR 比较。

梯度裁剪触发次数：AdamW **44/3052**、MuonW 0.01 **273/3052**、
MuonH 0.005/0.01/0.02 分别 **1332/576/819** 次、AdamH 三组分别
**106/56/48** 次。该指标记录裁剪前的全局梯度范数；高触发率值得保留，
但各完整组验证 loss 均继续下降，不能据此断定发散。短跑选 LR 也有
局限：AdamH 0.005 在第 1000/2000 步均优于 0.01，到终点却反转。
吞吐方面，MuonH 0.01 比 MuonW 0.01 再低约 **1.8%**，AdamH 0.01
比 AdamW 低约 **3.5%**。这些是同硬件的稳态训练速度比较；续跑生成的
`session_elapsed_seconds` 只覆盖最后一次会话，不能当作完整训练耗时。

## 复现与产物

在项目根目录运行。下例给出完整预算命令；将脚本、优化器、隐藏层 LR 和
输出目录按表替换，即可复现其余完整组。重跑须用新的输出目录，不能覆盖
已有 `runs/`。0.005/0.01 组先加 `--stop-after-steps 1000` 暂停，之后
保留原参数、去掉该参数并加 `--resume runs/<运行名>/checkpoint.pt` 续跑；
暂停不改变 3052 步 LR 计划。0.02 的 Hyperball 组直接跑完。

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_muon.py --data-dir data/tokenized/m01_fineweb_100m \
  --model-config configs/model/p099m.yaml --output-dir runs/m06_99m_muon_lr0p01 \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --muon-lr 0.01 --adamw-lr 1e-3 --warmup-steps 300 --min-lr-ratio 0.1 \
  --seed 2026 --eval-every 500 --save-every 1000 --log-every 100

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/pretrain_hyperball.py --optimizer muonh \
  --data-dir data/tokenized/m01_fineweb_100m \
  --model-config configs/model/p099m.yaml --output-dir runs/m06_99m_muonh_lr0p01 \
  --device cuda --precision bf16 --batch-size 4 --grad-accum-steps 2 --epochs 1 \
  --hyperball-lr 0.01 --adamw-lr 1e-3 --warmup-steps 300 --min-lr-ratio 0.1 \
  --seed 2026 --eval-every 500 --save-every 1000 --log-every 100
```

AdamH 使用第二条命令的 `--optimizer adamh` 和对应输出目录；各组
`--hyperball-lr` 分别取 0.005、0.01、0.02。独立验证统一执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python scripts/eval/base_eval.py loss \
  --checkpoint runs/m06_99m_muonh_lr0p01/checkpoint.pt \
  --data-dir data/tokenized/m04_fineweb_1b --device cuda:0 --precision bf16 \
  --batch-size 4 --output runs/m06_eval/val_m06_99m_muonh_lr0p01.json
```

把命令中的运行名替换为其他组即可。MuonW 与六组 Hyperball 的独立
评测 JSON 在 `runs/m06_eval/`；AdamW 基线评测在
`runs/m04_eval/val_99m.json`。训练 checkpoint、summary 和
TensorBoard 日志位于各自的 `runs/` 目录。

## 后续

保留 MuonW 的明确 loss/吞吐权衡及 Hyperball 的范数约束结果，不为
0.008–0.021 的单 seed 差值追加无计划的 M6 搜索。按 README 的 M7 目标，
下一步实现可测试的 top-k router 与专家层，再比较匹配计算量的 dense
基线和 MoE。
