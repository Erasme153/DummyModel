# DummyM

DummyM 是一个面向初学者的、从零实现并预训练 Llama-like Decoder-only 语言模型的学习型工程。项目以原生 PyTorch 为核心，目标是在两张 NVIDIA H20 或等价算力的 GPU 上跑通模型与 Tokenizer 实现、数据工程、预训练、scaling、分布式训练、评测、后训练和推理流程。

> 当前状态：早期开发阶段（alpha，M0 进行中）。miniLLaMA 模型、随机权重生成链路、模型单元测试和 tiny-corpus overfit 已经实现；自训练 Tokenizer、正式预训练数据流水线、TorchTitan/FSDP2、正式评测与后训练仍在规划或开发中。

## 项目定位与 Marin 的关系

DummyM 参考 [Marin](https://github.com/marin-community/marin) 开放代码、模型和实验结论的思路，但不复刻其面向大规模研发的管理体系。这里使用 PyTorch、TorchTitan 和两张 H20 或等价算力 GPU，让学习者能够读懂并亲手实现每一层。

### 技术栈差异

Marin 是覆盖数据、训练、评测和产物管理的研发框架，其语言模型训练主要由 Levanter 执行；DummyM 当前则自行组织这些流程，并直接使用 PyTorch 实现模型与训练。

| 方面 | Marin / Levanter | DummyM / PyTorch |
| --- | --- | --- |
| 核心计算 | JAX + XLA 编译 | PyTorch eager，后续可选 `torch.compile` |
| 模型与张量 | Equinox + Haliax 具名张量 | `nn.Module` + 普通 Tensor |
| 分布式 | JAX mesh，按具名轴组织 FSDP/TP | TorchTitan、FSDP2 和 PyTorch DeviceMesh |
| 主要优势 | 大规模 TPU/GPU 训练、静态图优化、实验与产物复现 | NVIDIA GPU 生态成熟、逐层调试直观、易接入 TRL 和 vLLM |
| 主要代价 | JIT 编译和函数式编程门槛较高 | 分片、恢复和实验记录需要自行实现与验证 |

DummyM 选择 PyTorch，是为了在两张 H20 上优先学习模型数学、训练循环和分布式基础。当前阶段只记录复现命令、关键参数、结果和结论；真正出现多组对比或大规模训练后，再引入更复杂的配置与追踪工具。

本项目采用以下原则：

- **先跑通，再扩展**：每个新功能先用小模型、小数据和单卡验证。
- **一项实验，一份说明**：在 `experiments/` 下用一个 README 记录目的、命令、结果和结论。
- **脚本保存过程，README 保存结论**：checkpoint、summary 和 TensorBoard 日志留在本地 `runs/`。

## 当前实现

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| Llama-like Decoder | 已实现 | RMSNorm、GQA causal attention、RoPE、SwiGLU MLP、残差连接和 LM Head |
| Attention backend | 已实现 | 使用 PyTorch SDPA，由 PyTorch 根据运行环境选择可用后端 |
| 随机权重推理 | 已实现 | 可借用本地 Hugging Face `tokenizer.json` 完成 prompt → token → logits → token → text 的冒烟测试 |
| 模型单元测试 | 已实现 | 覆盖 RMSNorm、RoPE、模型前向传播和生成逻辑 |
| Scaling 配置 | 仅有骨架 | `v001` 当前只有 39M 与 1.15B 目标占位文件，具体维度和中间档位尚未确定 |
| Tokenizer 训练 | 待实现 | 计划使用 Hugging Face Tokenizers 自行训练 BPE |
| 数据流水线 | 待实现 | 计划使用 Hugging Face Datasets 与 DataTrove |
| 预训练与分布式 | 部分实现 | tiny-corpus AdamW 训练已跑通；通用单卡 trainer、TorchTitan 与 FSDP2 待实现 |
| 评测、后训练和部署 | 待实现 | 计划分别接入 lm-evaluation-harness、TRL 和 vLLM |

## 模型结构

```text
Tokenization（Mistral-7B-v0.1，未来替换为自主训练的Tokenizer）
      ↓
Token Embedding
      ↓
N × Transformer Block
    ├─ RMSNorm
    ├─ GQA Causal Attention
    │    └─ RoPE
    ├─ Residual
    ├─ RMSNorm
    ├─ SwiGLU MLP
    └─ Residual
      ↓
RMSNorm
      ↓
LM Head
      ↓
Vocabulary logits
```

核心代码位于 [`src/dummym/models/llama_like`](src/dummym/models/llama_like)。当前实现用于理解和验证模型原理，还没有经过大规模训练正确性与性能验证。

## 环境安装

环境要求：Python 3.10 或更高版本、PyTorch 2.2 或更高版本。推荐为项目创建
独立 Conda 环境；下面的 `dummym` 只是示例名称，可以自行替换。

```bash
cd /path/to/pretrain
conda create --name dummym python=3.10 -y
conda activate dummym
python -m pip install --upgrade pip
```

NVIDIA GPU 用户先打开 [PyTorch Start Locally](https://pytorch.org/get-started/locally/)，
根据操作系统、Python 和驱动支持情况选择合适的 CUDA 版本，并执行页面生成的
安装命令。CPU 环境可以直接安装 CPU wheel：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

PyTorch 安装完成后，安装 DummyM 及测试依赖：

```bash
python -m pip install -e ".[dev]"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pytest tests/unit/model -q
```

以后重新进入项目时只需要激活已经创建的环境：

```bash
conda activate dummym
cd /path/to/pretrain
```

### 随机权重生成冒烟测试

下面的命令只验证端到端生成链路。模型参数是随机初始化的，借用现有开源模型的 Tokenizer，因此输出没有语言意义。

```bash
python scripts/inference/random_prompt_demo.py \
  "你好，请介绍一下你自己。" \
  --tokenizer /path/to/tokenizer.json \
  --device cpu \
  --max-new-tokens 16
```

H20 环境可将 `--device cpu` 改为 `--device cuda`。脚本默认路径只适用于当前开发机，公开仓库中的用法应始终显式传入 `--tokenizer`。

### Tiny-corpus overfit

这个 M0 实验使用现成的 Mistral 32K Tokenizer，将 [`data/tiny_corpus.txt`](data/tiny_corpus.txt) 的每一行编码后追加 EOS，再拼成一个固定的 `4 × 128` token batch。训练过程始终重复这个 batch，用来检查 tokenization、causal LM loss、反向传播、AdamW 更新和 checkpoint 保存是否能够连通；它不衡量模型的泛化能力。

```bash
python scripts/train/tiny_overfit.py \
  --tokenizer /path/to/Mistral-7B-v0.1/tokenizer.json \
  --device cuda
```

查看训练曲线：

```bash
tensorboard --logdir runs/m00_tiny_overfit/tensorboard
```

2026-09-03 使用 Python 3.10 和 PyTorch 2.5.1+cu124 在单张 NVIDIA H20 上验收（seed 2026）：loss 从 `10.381273` 降至 `0.049335`，第 223 step 达到停止条件，next-token accuracy 为 `1.0000`。训练摘要和 checkpoint 分别保存为 `runs/m00_tiny_overfit/summary.json` 与 `runs/m00_tiny_overfit/checkpoint.pt`；`runs/` 是本地运行产物，不提交 Git。

完整实验结论见 [`experiments/m00_foundations/exp001_tiny_overfit/README.md`](experiments/m00_foundations/exp001_tiny_overfit/README.md)。

## 模型规模与 Scaling 约定

项目中有两类模型规模，不应混为一谈：

- **教学锚点**：39M 用来打通端到端预训练，99M 用来学习超参数 sweep，213M 用来产出第一版正式 base model。
- **Scaling ladder 候选点**：用于拟合 scaling law、选择给定算力预算下的模型/数据组合，并验证对更大规模的预测。

当前 `v001` 暂时保留以下近似等比候选规模：

```text
p039m → p077m → p151m → p297m → p584m → p1150m
```

这些尺寸目前只是草案。[`configs/model/ladder/v001`](configs/model/ladder/v001)
中的配置也仍是占位文件，等 39M/99M 教学实验产生真实数据后再补全，不提前
建立版本冻结、run ID 等管理规则。

这里仍区分两个问题：**scaling law** 研究给定算力下模型大小和训练 token 数，
**training recipe** 研究 learning rate、batch size 和 schedule。真正开始 scaling
实验时再为批量运行增加配置文件。

## 计划采用的技术栈

| 环节 | 计划方案 |
| --- | --- |
| 模型与基础训练 | 原生 PyTorch |
| 双卡与大模型训练 | TorchTitan + FSDP2 |
| Attention | PyTorch SDPA / Flash Attention backend |
| 数据处理 | Hugging Face Datasets + DataTrove |
| Tokenizer | Hugging Face Tokenizers，自训练 BPE |
| 实验记录 | W&B 或 TensorBoard |
| 预训练评测 | lm-evaluation-harness |
| 后训练 | TRL |
| 推理与快速评测 | vLLM |
| 性能分析 | `torch.profiler` + Nsight Systems |
| Scaling 分析 | NumPy、SciPy、pandas、matplotlib |

除 PyTorch 和 Hugging Face Tokenizers 外，上表多数依赖尚未加入项目依赖，也未完成集成。

## 数据方案（草案）

候选语料配比为 80% FineWeb-Edu、10% The Stack v2 deduplicated 和 10% OpenWebMath。该比例尚未冻结；下载或训练前还必须核对每个数据源的许可证、访问条件、字段格式、去重范围和实际 token 统计。

```text
Hugging Face Datasets streaming
              ↓
            抽样
              ↓
          DataTrove
              ↓
过滤 → 规范化 → 去重 → 质量检查
              ↓
          自训练 Tokenizer
              ↓
      EOS 拼接与定长 packing
              ↓
        2048-token shards
              ↓
          本地 NVMe
```

原始数据、处理中间产物、tokenized shards、checkpoint 和运行日志不提交 Git。
真正开始使用外部数据时，再补一份简短的数据来源、许可证和处理说明。

## 实验记录

每项实验在 `experiments/<阶段>/<实验名>/README.md` 中记录：

- 目的和验收条件；
- 数据、模型和训练的关键参数；
- 可直接执行的命令；
- 实际结果；
- 结论和下一步。

默认不创建实验 YAML、独立 results 文件、run ID 或配置快照。脚本能够清楚表达
的参数就留在脚本和命令行中；只有开始多组 sweep 时才引入 YAML。checkpoint、
summary 和 TensorBoard 日志由脚本写入 `runs/<实验名>/`。

## 目录结构

```text
pretrain/
├── configs/          # 需要复用或批量运行时才增加的配置
├── experiments/      # 每项实验一份 README
├── data/             # 小型测试语料及未来的数据处理目录
├── scripts/          # 训练、推理和数据处理入口
├── src/dummym/       # 可复用 Python 包源码
├── tests/            # 自动化测试
├── runs/             # 本地 checkpoint、summary 和 TensorBoard 日志
└── pyproject.toml    # Python 包、依赖与测试配置
```

## 简化的工作方式

- 代码放在 `src/` 和 `scripts/`，保持可以直接运行。
- 每项实验只保留一个 README，成功和失败都写在同一处。
- 自动生成的训练产物放在 `runs/`，不提交 Git。
- 当参数组合明显增多时，再增加配置文件和实验追踪工具。

当前执行顺序：

```text
实现脚本 → 小数据运行 → 检查结果 → 在实验 README 写结论 → 继续下一步
```

## 学习路线与里程碑

里程碑按学习依赖排列。每一阶段至少留下可运行代码、测试和一份简短实验结论。

| Milestone | 要真正学会什么 | 主要交付物与通过条件 |
| --- | --- | --- |
| **M0 · Foundations** | 自己实现 Transformer 与 BPE Tokenizer | 完成 RMSNorm、RoPE、GQA、SwiGLU、causal loss、采样生成；通过 shape、mask、数值测试和 tiny-corpus overfit。当前 tiny-overfit 已通过，自训练 Tokenizer 待完成。 |
| **M1 · 39M from scratch** | 第一次完整预训练，而不是只会调用 Trainer | 完成数据清洗、tokenization、packing、train/validation split、单卡训练、验证以及 checkpoint 保存与恢复。 |
| **M2 · 99M recipe sweep** | 学会控制变量和选择训练 recipe | 在固定模型、数据、token budget 和随机种子策略下，对 LR、AdamW betas/weight decay、warmup/decay schedule 做 sweep；按预先声明的验证 loss、稳定性与吞吐指标选择 recipe，并保留失败实验。新型优化器研究留到 M6，避免变量混杂。 |
| **M3 · Distributed systems** | DDP、TorchTitan/FSDP2 与 profiling | 对齐单卡和双卡的首步/短程 loss；验证梯度累积、混合精度、分布式 checkpoint 与恢复；用 `torch.profiler`/Nsight 分析吞吐、显存和通信瓶颈。此阶段先做 dense data parallel，EP 留到 M7。 |
| **M4 · 213M base pretraining** | 运行第一版“正式”base model 训练 | 确定 Tokenizer、数据和训练方法；在两张 H20 上完成可恢复训练，产出 checkpoint、训练报告、base eval 和模型卡。 |
| **M5 · Mini-Delphi scaling** | IsoFLOP、scaling law、scaling recipe 与外推验证 | 设计多个 compute budget 和候选 `(参数量, token 数)`；小规模点使用重复 seed；拟合并报告不确定性；用 held-out 规模检验预测，再决定是否运行最高至 1.15B 的 ladder。`Mini-Delphi` 是 DummyM 的教学实验名，不代表 Marin 官方 Delphi 的复现结果。 |
| **M6 · Optimizer research** | Muon 与 Hyperball 系列方法如何公平比较 | 以 M2/M5 的 AdamW recipe 为固定基线，对 Muon 及 Hyperball 约束版本（如 AdamH/MuonH）做单变量 A/B；记录 loss、吞吐、参数范数和 update/parameter ratio；新优化器使用独立 scaling heuristic，不能直接沿用 AdamW 最优参数。 |
| **M7 · Mini-MoE systems** | Sparse MoE、Router、负载均衡和 Expert Parallel | 先实现可测试的 top-k router 与专家层，再加入容量、token dispatch/combine、负载与丢 token 指标；将 Quantile Balancing（QB）作为独立路由实验；与 active-parameter/compute 匹配的 dense baseline 比较，最后接入 EP 并 profile 通信。 |
| **M8 · Midtraining & cooldown** | 数据混合变化与学习率退火如何影响能力 | 从同一 base checkpoint 分叉，对高质量/领域数据配比、阶段 token budget 和 cooldown schedule 做受控实验；同时看通用能力保持、目标能力增益和遗忘，而不只看单项 benchmark。 |
| **M9 · Post-training** | SFT、偏好优化与可验证奖励强化学习 | 定义 chat template 和数据审计；完成 SFT、DPO、GRPO/RLVR 的目标函数、reference policy/奖励与评测。学习顺序可按 SFT → DPO → GRPO，但默认应从同一 SFT checkpoint 建立 DPO 与 GRPO 对照分支，不假设三者必须串行才正确。 |

### 每个里程碑的统一完成标准

一个 milestone 标记完成前，至少满足：

1. 代码可以重新运行，并通过相关测试。
2. 关键结果符合预期，失败情况也有说明。
3. checkpoint 等必要产物可以加载。
4. 实验 README 记录了命令、结果、结论和下一步。

## 参考项目与方法

- [Marin](https://github.com/marin-community/marin)：开放记录数据、训练、评测、实验决策与失败结果的方法论参考。
- [Marin: Train a language model](https://marin.readthedocs.io/en/latest/tutorials/train-an-lm/)：从数据混合、模型配置到训练任务的完整示例。
- [Marin scaling heuristic recipe](https://github.com/marin-community/marin/blob/main/docs/recipes/add_scaling_heuristic.md)：区分 scaling law 与 training heuristic，并组织 IsoFLOP sweep。
- [Delphi scaling suite](https://github.com/marin-community/marin/issues/1337)：开放 scaling suite、重复 seed、统一评测和可复现数据顺序的参考。
