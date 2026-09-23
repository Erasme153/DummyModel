# M1：39M 英文预训练基线

## 目的与范围

从随机初始化开始，跑通真实语料的数据准备、单卡训练、独立验证和 checkpoint
保存/恢复。**本轮 M1 已完成并通过流程验收**：正式运行在第 100 步暂停后恢复，
最终完成 3052 次更新和 99,999,744 个输入 tokens；全量验证 loss 为 4.153138。
最终 checkpoint 可严格加载并生成文本。生成仍有重复与语义不连贯，这一能力
限制保留在报告中，不将“流程通过”等同于“语言能力合格”。

| 项目 | 首轮约定 |
| --- | --- |
| 模型 | MiniLlama：8 层、hidden=512、MLP=1408、Q heads=8、KV heads=2、共享 embedding，实际 38,937,088 参数 |
| 设备 | 单张 NVIDIA H20，BF16 autocast，参数与 AdamW 状态保持 FP32 |
| 数据 | 100% 本地 FineWeb-Edu 英文文本，不混入代码或数学数据集 |
| Tokenizer | 现成的 Mistral-7B-v0.1 32K tokenizer；本轮固定不变 |
| 文档边界 | 正文编码时关闭自动特殊 token，末尾追加 EOS，不添加 BOS |
| 序列长度 | 2048，不补 padding，允许一行包含多篇文档 |
| 训练数据预算 | 100,000,000 tokens，向下对齐为 99,999,744 tokens |
| 验证数据预算 | 1,000,000 tokens，向下对齐为 999,424 tokens |
| 划分 | 规范化正文的 SHA-256 指纹决定 split，验证文档概率 1% |
| 随机种子 | 2026 |
| 有效 batch | micro-batch 4 条序列 × 梯度累积 4 次 = 16 条，通常每次更新输入 32,768 tokens |
| 优化器 | AdamW，betas=(0.9, 0.95)，矩阵 weight decay=0.1，RMSNorm 不衰减，梯度裁剪 1.0 |
| 学习率 | 峰值 3e-4，100 次更新 warmup，之后 cosine 衰减至峰值的 10% |
| 完整计划 | 1 epoch，共 3052 次 optimizer update，尾部 12 条序列也参与训练 |
| 记录 | 一份实验 README；数据摘要、checkpoint、TensorBoard 留在本地 |

以上预算包含 EOS，指写入序列的 token 数；当前模型每行内部预测后 2047 个
token，所以 loss 的有效预测位置数与输入 token 数不同。首轮训练暂按约一遍
训练数据安排，不将多次重复小 batch 当作完成正式预训练。

继续使用现成 tokenizer 是本轮的教学安排；README 中的自训练 BPE 仍是 M0
待办。完成它后，应另开 tokenizer 对比实验并重新编码数据，不在本轮中途换词表。

## 数据来源和处理

来源：[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。
本地数据卡声明许可证为 `odc-by`。原始 `url` 和文档 `id` 保存在处理索引中，
方便追溯来源；数据集声明不等于所有原始网页采用相同许可证。

当前下载目录为项目外的 `../data/fineweb-edu`。只读取其中
`data/CC-MAIN-*/*.parquet`，排除尚未完成的文件和 `sample/` 抽样副本。
准备脚本按固定种子打乱每个抓取批次的分片、row group 及 batch 内文档，再在
不同抓取批次间轮询。**这是对本地数据的抽样，不是全库均匀随机样本。**

处理规则保持简单，全部写在脚本中并带中文注释：

1. 使用 PyArrow 分批读取 `text`、`id`、`url`，不把整个分片加载到内存。
2. Unicode NFC 规范化，统一换行和空白，保留段落、大小写及标点。
3. 过滤非字符串、控制字符、乱码替换字符；保留 200～100,000 字符且
   字母占比至少 50% 的文本。这是轻量规则，不是新的语言或质量分类器。
4. 在所扫描的数据范围内，用规范化正文 SHA-256 做精确去重。
5. 按文档指纹和种子分配 train/validation，同样正文始终进入同一 split。
6. 分别编码、追加 EOS 和 packing；达到各自预算后停止收集对应 split。

不做全库近似去重，也不保证同一网站或部分重合的段落不会跨 split。验证集衡量
当前本地语料分布上的 next-token loss，不代表完整 FineWeb-Edu 或通用能力评测。

## 准备命令

在项目根目录、已激活项目 Python 环境后执行。依赖由 `.[data]` 声明；本次开发机
已具备依赖，没有为本实验安装、升级包或新建环境。

```bash
PYTHONNOUSERSITE=1 python scripts/data/prepare_fineweb.py \
  --input-dir ../data/fineweb-edu \
  --tokenizer /path/to/Mistral-7B-v0.1/tokenizer.json \
  --output-dir data/tokenized/m01_fineweb_100m \
  --train-tokens 100000000 \
  --validation-tokens 1000000 \
  --sequence-length 2048 \
  --seed 2026
```

`--tokenizer` 替换为现有本地文件。源数据目录变化时同步修改 `--input-dir`。
命令不联网，所有参数都由命令行控制，无需额外 manifest 或实验 YAML。

输出目录必须不存在，避免覆盖已有结果。异常退出会保留现场，重新运行时选择
一个新输出目录；本脚本暂不支持从数据处理断点恢复。只有 `summary.json` 中
`status` 为 `complete`，且文件长度与摘要一致，才说明预算已经准备完成。

## 数据产物与读取

```text
data/tokenized/m01_fineweb_100m/
├── train.bin          # 连续的小端 uint16 token ID
├── validation.bin     # 完全独立的验证 token 流
├── tokenizer.json     # 实际用于编码的 tokenizer 副本
├── documents.jsonl    # split、正文指纹、来源、token 位置及贡献量
├── preview.jsonl      # 每个 split 前三篇入选文档的文本预览
└── summary.json       # 完成状态、参数、来源、过滤计数、tokenizer 指纹及 token 数
```

这些文件全部在 Git 忽略的目录中，不提交原始文本、token 数据或处理索引。
一个二进制文件按每 2048 个 ID 解释为一行，不包含文件头或 padding。

```python
import json
from pathlib import Path

import numpy as np
import torch

directory = Path("data/tokenized/m01_fineweb_100m")
summary = json.loads((directory / "summary.json").read_text())
assert summary["status"] == "complete"
length = summary["format"]["sequence_length"]
tokens = np.memmap(directory / "train.bin", dtype="<u2", mode="r")
assert tokens.size == summary["splits"]["train"]["written_tokens"]
sequences = tokens.reshape(-1, length)
input_ids = torch.from_numpy(sequences[:4].astype(np.int64))
# 训练时调用 model(input_ids=input_ids, labels=input_ids)。
# MiniLlama 内部用 logits[:, :-1] 预测 labels[:, 1:]，不要在数据端再做一次 shift。
```

EOS 只标记文档边界；本轮使用普通 causal attention，不额外隔离同一行中的文档。
最后一篇文档可能在预算边界被截断，截断数记录在摘要中。

## 数据准备结果（2026-09-12）

使用现有环境完成处理，耗时 **117.24 秒**。发现 134 个本地正式分片，实际轮询
读取 10 个抓取批次中的 10 个分片就达到预算，来源为 2013～2014 年的本地数据。

| 数据 | 文档数 | 2048-token 序列数 | 实际 tokens | 二进制字节数 |
| --- | ---: | ---: | ---: | ---: |
| train | 89,141 | 48,828 | 99,999,744 | 199,999,488 |
| validation | 905 | 488 | 999,424 | 1,998,848 |

共扫描 90,496 篇文档，过滤过长文本 95 篇、含乱码替换字符文本 74 篇、低字母占比
文本 1 篇，去除规范化后完全重复的文本 183 篇；另有 97 篇因其 split 已达到预算
未被使用。两个 split 最后一篇的预算截断分别舍弃 84、1,969 个 tokens。

实际 tokenizer 副本 SHA-256：
`835cf54bb933752267652500a9243a9002107bbb4501b99f8d2f46ccfdad2567`。
详细来源文件、每个来源的入选数、参数和统计保存在本地 `summary.json`。

已完成的检查：

- 扫描全部生成 token，均在 `[0, 32000)` 范围内，两个文件字节数、序列数与摘要一致。
- 核对全部 90,046 个文档指纹，没有 split 内精确重复或跨 split 精确重叠；文档划分
  符合固定种子的哈希规则，token 偏移连续，未截断文档的最后一个 token 均为 EOS。
- 解码首段 token，可以还原正常英文文本；将一个完整 2048-token 序列传入现有
  MiniLlama 小配置，forward 和 causal loss 正常。这只是接口验证，不是 39M 训练结果。
- 数据回归测试覆盖清洗、EOS/拼接顺序、重复运行一致性、分片间去重、损坏格式报错、
  预算不足报错及拒绝覆盖。与已有模型测试一起运行：**19 passed**。

检查命令：

```bash
PYTHONNOUSERSITE=1 python -m pytest tests/unit/model tests/unit/data -q
```

结论：**数据准备通过，可以进入模型配置和 trainer 实现。**

## 验收标准

数据准备检查：文件可读、token ID 在词表范围内、长度为 2048 的整数倍、
文档指纹不重复且训练/验证无精确正文重叠、同参数同输入可重复生成同样内容。

训练阶段以 loss 有限且下降、独立验证 loss 改善、恢复检查通过和最终 checkpoint
可加载为准，不以训练 loss 接近 0 或具备聊天能力为验收条件。当前完整训练、
回归检查及生成测试结果见下文“完整训练与最终验收”。

## 单卡训练与恢复

入口为 [`scripts/train/pretrain.py`](../../../scripts/train/pretrain.py)，模型架构直接读取
[`p039m.yaml`](../../../configs/model/ladder/v001/p039m.yaml) 的完整 `model_config`。
不做 YAML 递归继承，也不额外创建训练配置文件。新增依赖由 `.[train]` 声明；
本次使用现有 Python 3.10、PyTorch 2.5.1+cu124 环境，没有安装或升级依赖。

在项目根目录执行一个新的 100 步短跑：

```bash
PYTHONNOUSERSITE=1 python scripts/train/pretrain.py \
  --device cuda:1 \
  --output-dir runs/m01_39m \
  --stop-after-steps 100
```

`cuda:1` 表示当前进程可见的第二张 GPU，可按实际空闲设备改为 `cuda:0`。
若启动命令前设置了 `CUDA_VISIBLE_DEVICES=1`，进程只能看到原来的 1 号卡，
但它在进程内重新编号为 `cuda:0`，此时必须传 `--device cuda:0`（或 `cuda`）。
例如 `CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python scripts/train/pretrain.py
--device cuda:0 --stop-after-steps 100`。不要同时设置单卡可见性和 `--device cuda:1`。
脚本会在启动时校验逻辑编号，并打印可见卡数、选择的设备和可见性变量；修改
`CUDA_VISIBLE_DEVICES` 后需重新启动 Python/调试进程。这遵循
[CUDA 的设备枚举规则](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。

不传 `--stop-after-steps` 会执行完整 1 epoch。**短跑停止点不会把学习率计划压缩
成 100 步**：上述命令仍按完整 3052 步计算调度，只是在第 100 步保存后暂停。
参数是“总更新步数”，恢复时设为 200 表示跑到第 200 步，并非再跑 200 步。

从刚才的短跑继续完成本轮训练：

```bash
PYTHONNOUSERSITE=1 python scripts/train/pretrain.py \
  --device cuda:1 \
  --resume runs/m01_39m/checkpoint.pt
```

恢复时默认把输出写回 checkpoint 所在目录；也可以用 `--output-dir` 指定一个
尚不存在的新目录，保留原 checkpoint 作对照。新训练拒绝覆盖已有输出目录。

恢复会核对模型配置、两个二进制数据文件和 tokenizer 的内容指纹，以及 batch、
LR、epochs、precision 等训练约定。自定义过这些参数时，恢复命令也应保持一致。
可以改变设备序号、打印/验证/保存间隔和暂停步数；不能把 BF16 切成 FP32 后声称
是相同训练过程。数据顺序由 seed+epoch 重建，再从保存的序列位置继续。

默认初始、每 100 步及本次结束时验证全部 488 条验证序列。调试可用
`--eval-batches N` 限制为固定前 N 个 batch，但那不是完整验证集指标，恢复时也
必须保持这个设置不变。CPU 调试需要 `--device cpu --precision fp32`，建议另用
小模型、小数据，并将 `--warmup-steps` 调整到小于完整训练步数。

TensorBoard：

```bash
tensorboard --logdir runs/m01_39m/tensorboard
```

记录 loss、实际使用的 LR、裁剪前梯度范数、tokens/s、累计输入 tokens、GPU
峰值 allocated 显存以及验证 loss。tokens/s 只计训练更新，不含验证与写盘时间；
显存数值是本进程 PyTorch 分配峰值，不是整张 GPU 上所有进程的显存占用。

本地输出保持简单：`checkpoint.pt`、`summary.json`、`tensorboard/`。
checkpoint 含模型、AdamW、scheduler、CPU/CUDA 随机状态、step、epoch 和下一个
读取位置；只在完整 optimizer update 后保存。遇到异常或 Ctrl+C，从最近保存点
恢复，未保存的更新会重做。`summary.json` 在正常暂停/完成时写出，异常后应以
实际 checkpoint 的进度为准，不能把旧 summary 当成最新进度。

训练 checkpoint 没有 tiny-overfit 的 `corpus_path` 字段，不适用带近零 loss 验收的
`tiny_checkpoint_demo.py`。使用通用入口
[`pretrained_checkpoint_demo.py`](../../../scripts/inference/pretrained_checkpoint_demo.py)
从 `model_config`、`model_state_dict` 和 `tokenizer_path` 重建模型，命令见下文。

## 训练脚本验证（历史短跑，2026-09-14）

在单张 H20 上使用上述默认训练参数，日志和 checkpoint 放在
`runs/m01_39m_smoke/`；本次短跑额外设置 `--eval-every 50 --save-every 50`。
短跑 100 次更新，共读取 3,276,800 个训练输入 tokens，仍未完成 1 亿 token 预算。
这是 trainer 开发阶段的历史记录；当前本地已无该 smoke 目录。以下历史命令仅用于
说明当时的验证方法，不保证原路径仍可执行。最终模型以 `runs/m01_39m/` 为准。

| 指标 | 实测值 |
| --- | ---: |
| 第 1 步训练 loss | 10.467550 |
| 第 100 步训练 loss | 6.956877 |
| 初始全量验证 loss | 10.473185 |
| 第 100 步全量验证 loss | 6.948971 |
| 第 1 / 第 100 步 LR | 3e-6 / 3e-4 |
| 稳定更新吞吐 | 约 169,000 输入 tokens/s，不含验证与写盘 |
| PyTorch 峰值 allocated 显存 | 约 5.47 GiB |
| 本次训练会话耗时 | 31.16 秒，包含会话内验证与保存，不含启动加载 |

另从第 100 步 checkpoint 恢复到第 102 步，输出至独立的
`runs/m01_39m_resume_check/`，保留原短跑 checkpoint。恢复后的下一个序列位置为
1632，累计输入 tokens 为 3,342,336；LR 延续原来的 3052 步计划，全量验证 loss
为 **6.917020**。本次恢复检查命令：

```bash
PYTHONNOUSERSITE=1 python scripts/train/pretrain.py \
  --device cuda:1 \
  --resume runs/m01_39m_smoke/checkpoint.pt \
  --output-dir runs/m01_39m_resume_check \
  --stop-after-steps 102 \
  --eval-every 50 --save-every 50
```

新增训练回归测试与已有模型、数据测试共 **24 passed**。CPU 小模型测试在启用
dropout、存在不足 batch 的尾部并跨越 epoch 时，对比不中断训练与中断后恢复：
模型参数、AdamW 状态、scheduler、游标和 RNG 均完全一致。还验证了数据内容改变
时拒绝恢复、验证集尾部加权、梯度累积与等效大 batch 的更新一致性。

```bash
PYTHONNOUSERSITE=1 python -m pytest tests/unit -q
```

这些结果验证的是脚本正确性与短程收敛，不保证不同硬件、PyTorch 版本或 CUDA
算子选择之间逐位复现，也不代表模型已经具备良好的通用语言能力。

## 完整训练与最终验收（2026-09-22 整理）

本节以正式运行的 `runs/m01_39m/summary.json`、两个 TensorBoard 事件文件和
最终 `checkpoint.pt` 为依据；与上面的历史 smoke run 分开记录。

正式运行先暂停于 step 100，再从同一目录的 checkpoint 恢复到 step 3052。
两个事件文件分别记录 step 1～100 和 step 101～3052，合并后共有 3052 条训练
loss 记录、32 条全量验证 loss 记录。以下命令描述本轮运行过程；已有结果目录
不能用于重新开始训练，如需复跑请使用另一个 `--output-dir`。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
python scripts/train/pretrain.py \
  --device cuda:0 \
  --output-dir runs/m01_39m \
  --stop-after-steps 100

CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
python scripts/train/pretrain.py \
  --device cuda:0 \
  --resume runs/m01_39m/checkpoint.pt
```

### 最终指标

| 指标 | 正式运行结果 |
| --- | ---: |
| 完成状态 | `complete` |
| 参数量 | 38,937,088 |
| epoch / optimizer updates | 1 / 3052 |
| 输入 tokens / 有效预测 tokens | 99,999,744 / 99,950,916 |
| 初始全量验证 loss | 10.473185 |
| 最终全量验证 loss | 4.153138 |
| 最终验证 perplexity，`exp(loss)` | 63.6333 |
| 第一 / 最后一个训练 batch 的 loss | 10.467550 / 4.103119 |
| 每次验证序列数 / 有效预测位置数 | 488 / 998,936 |
| 首步 / warmup 峰值 / 最后一步 LR | 3e-6 / 3e-4 / 3e-5 |
| 裁剪前梯度范数最大值 / 最终值 | 3.405901 / 0.912229 |
| 梯度范数大于裁剪阈值 1.0 的更新数 | 187 / 3052 |
| 稳定更新吞吐中位数 | 约 167,608 输入 tokens/s |
| PyTorch 峰值 allocated 显存 | 5.468 GiB |
| 恢复后会话耗时，step 101～3052 | 676.21 秒 |

吞吐中位数取 step ≥ 10 且排除恢复后的首步 101，只计训练更新，不含验证和
checkpoint 写盘。676.21 秒是摘要记录的**恢复后会话**耗时，不能当作两次会话
的总训练时间。显存为 PyTorch allocated 峰值，不是 `nvidia-smi` 的总显存占用。

训练 loss 是各更新使用的 batch 在更新前的平均值，最终的 4.103119 不是全训练集
评估值；不能仅据其与验证 loss 的差值精确衡量泛化差距。Perplexity 只在相同
tokenizer、验证数据和计分约定下可比。

| Step | 全量验证 loss |
| ---: | ---: |
| 0 | 10.473185 |
| 100 | 6.949314 |
| 500 | 5.298059 |
| 1000 | 4.751925 |
| 1500 | 4.490235 |
| 2000 | 4.307721 |
| 2500 | 4.205183 |
| 3000 | 4.156441 |
| 3052 | 4.153138 |

全部 32 个验证点持续下降，记录的训练梯度范数均有限；没有观察到验证 loss
反弹的过拟合迹象。训练末期仍有改善，但这不能单独证明已充分收敛，也不能证明
训练代码完全无误。

### 最终 checkpoint 加载与生成

2026-09-22 使用现有 Python 环境补做固定 prompt 检查：全部样例采用 **CPU / FP32**，
严格加载返回 `<All keys matched successfully>`，读取到 step 3052 和 99,999,744
个训练输入 tokens。每次最多生成 32 个 token，不添加 BOS 或 chat template，
没有重复惩罚。这里的 seed 控制生成采样，与 checkpoint 的训练 seed 分开理解。

复现下面的采样基线（在项目根目录执行）：

```bash
PYTHONNOUSERSITE=1 python scripts/inference/pretrained_checkpoint_demo.py \
  --checkpoint runs/m01_39m/checkpoint.pt \
  --device cpu \
  --precision fp32 \
  --prompt "The future of artificial intelligence" \
  --max-new-tokens 32 \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.9 \
  --seed 2026
```

下表都是新增文本的实际摘录，省略其余 token；换用 GPU / BF16 时不要求生成
文本与这些 CPU / FP32 样例逐字一致。

| Prompt | temperature / top-k / top-p / seed | 新增文本摘录 |
| --- | --- | --- |
| The future of artificial intelligence | 0 / 0 / 1.0 / 2026 | `is the most important part of the world. The world is the world of the world.` |
| 同上 | 0.7 / 50 / 0.9 / 2026 | `, and the public, the scientific and scientific community of the United States` |
| 同上 | 0.7 / 50 / 0.9 / 2027 | `is the most important and best of the world's most famous` |
| 同上 | 1.0 / 0 / 0.95 / 2028 | `projects has provided the attempt to witness his knowledge in the artistic difficulty` |
| Water is important because | 0.7 / 50 / 0.9 / 2026 | `the area is not as important to the area as a consequence.` |
| The solar system consists of | 0.7 / 50 / 0.9 / 2026 | `the air and air. The air is the air and the air is the air pressure.` |

第一条摘录将换行展示为空格。`top-k=0` 表示禁用 top-k，`temperature=0` 时
top-k/top-p 不参与生成。不同 seed 得到了不同输出，说明采样起作用；随机采样
仍可能产生重复。更宽采样的这个例子重复减轻，但语义依然不连贯。它同时改变了
多个参数和 seed，仅作探索性样例，不能用于确定哪一个参数造成了变化。

结论：模型已学到部分英语词组和句式，但容易重复、偏离主题，并生成事实错误。
目前生成样例不足以判定重复的根因；训练预算、模型容量、数据特点和解码方式
都可能影响结果。不能把没有 repetition penalty 视为已经证实的根因，也不能
用降低重复后的表面流畅度替代语言能力评估。

### 验收结论与后续实验

**M1 流程验收通过，可以进入 M2。** 数据准备、完整预算训练、独立验证、
保存/恢复、最终模型加载和生成均已有检查记录。2026-09-22 复跑
`PYTHONNOUSERSITE=1 python -m pytest tests/unit -q`，结果为 **26 passed**。
其中 CPU 小模型中断恢复与不中断训练的参数/状态一致性已覆盖；实际 H20 运行
也从 step 100 恢复并完成预算，但本轮没有做完整 GPU 两条训练轨迹逐位一致的对照。

本阶段没有必须追加的大型训练实验。保留以下产物，报告与代码进入 Git，数据、
checkpoint 和事件日志按现有规则留在本地：

- `runs/m01_39m/checkpoint.pt`、`summary.json`、`tensorboard/`；
- `data/tokenized/m01_fineweb_100m/`，包含实际训练/验证 token 数据、tokenizer
  副本和数据摘要；模型恢复所需的数据指纹也保存在 checkpoint 中。

以下对照为可选扩展，确有对应问题时再开展：

| 要回答的问题 | 可选实验 | 边界 |
| --- | --- | --- |
| 39M 增加训练量后质量是否改善？ | 固定模型、tokenizer 和验证集，比较不同训练预算 | 优先使用更多独立训练文本；重复原数据需单独注明，扩充数据时保持验证隔离；当前恢复约定不允许直接改 epochs/数据后续跑 |
| 相同设置是否对随机种子敏感？ | 用另一个训练 seed 完整复跑一次 | 生成 seed 与训练 seed 不同；一次训练不能给出结果方差 |
| 解码参数对重复有多大影响？ | 固定 checkpoint、prompt 集和 seed 集，一次只改变一个参数 | 解码实验不证明模型语言能力提升 |

按主 README 的学习路线，下一阶段为 **M2：99M training recipe 对比**，先准备
模型配置和短跑验证，再在固定数据与预算下比较少量学习率候选。增加训练预算、
自训练 tokenizer、FSDP、正式 benchmark 或重复惩罚，都不作为本轮 M1 的补验收
前置条件。无需为本次结项新增实验 YAML、独立结果文件或新的管理目录。

## 实现参考

- [PyTorch 2.5.1 AMP examples](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)：autocast 范围、梯度累积以及裁剪时机。
- [PyTorch：学习率调度](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)：先执行 optimizer.step，再执行 scheduler.step。
- [PyTorch：序列化](https://docs.pytorch.org/docs/stable/notes/serialization.html)：保存 state_dict，使用受限的 weights_only 加载；本脚本在 PyTorch 2.5.1 上显式传入该参数。
