# DummyM

DummyM 是一个面向初学者的、从零实现并预训练 Llama-like Decoder-only 语言模型的学习型工程。项目以原生 PyTorch 为核心，目标是在两张 NVIDIA H20 上完整走通模型与 Tokenizer 实现、数据工程、预训练、scaling、分布式训练、评测、后训练和推理流程，而不只是得到一个最终 checkpoint。

> 当前状态：早期开发阶段（alpha，M0 进行中）。miniLLaMA 模型、随机权重生成链路和模型单元测试已经实现；自训练 Tokenizer、预训练数据流水线、训练循环、TorchTitan/FSDP2、正式评测与后训练仍在规划或开发中。本文会明确区分“已经实现”和“计划采用”，避免把路线图当作现有能力。

## 项目定位与 Marin 的关系

DummyM 参考 [Marin](https://github.com/marin-community/marin) 的开放研发方法：训练过程不仅公开最终代码和模型，还应保留实验假设、数据与配置、运行记录、失败结果和复盘材料。DummyM 不是 Marin 的复刻或精简分支，也不会照搬其 TPU/集群基础设施；这里使用 PyTorch、TorchTitan 和两张 H20，重点是让单个学习者能够读懂并亲手实现每一层。

本项目采用以下原则：

- **先正确，再扩展**：每个新功能先在小模型、小数据和单卡上验证，再进入多卡或更大规模。
- **实验先登记**：正式运行前写清假设、对照组、唯一变量、预算、指标和停止条件；运行后保留成功与失败结论。
- **配置即实验身份**：代码 commit、配置快照、数据 manifest、Tokenizer 哈希和随机种子共同决定一个 run。
- **产物有依赖关系**：数据、Tokenizer、checkpoint、评测和报告必须能追溯上游输入，不能只靠文件名猜测。
- **不同实验族不混比**：模型结构、Tokenizer、数据配比或优化器规则发生实质变化时，新建 recipe/ladder 版本。
- **先做基线再做研究特性**：AdamW dense baseline 未稳定前，不进入 Muon、Hyperball 或 MoE 对比。

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
| 预训练与分布式 | 待实现 | 计划先完成单卡 reference trainer，再接入 TorchTitan 与 FSDP2 |
| 评测、后训练和部署 | 待实现 | 计划分别接入 lm-evaluation-harness、TRL 和 vLLM |

## 模型结构

```text
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

## 快速开始

环境要求：Python 3.10 或更高版本、PyTorch 2.2 或更高版本。GPU 环境应先安装与机器 CUDA/驱动相匹配的 PyTorch，再安装本项目。

```bash
cd /path/to/pretrain
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest tests/unit/model -q
```

### 随机权重生成冒烟测试

下面的命令只验证端到端生成链路。模型参数是随机初始化的，因此输出没有语言意义；借用的是 Tokenizer，不会加载对应模型的权重。

```bash
python scripts/inference/random_prompt_demo.py \
  "你好，请介绍一下你自己。" \
  --tokenizer /path/to/tokenizer.json \
  --device cpu \
  --max-new-tokens 16
```

H20 环境可将 `--device cpu` 改为 `--device cuda`。脚本默认路径只适用于当前开发机，公开仓库中的用法应始终显式传入 `--tokenizer`。

## 模型规模与 Scaling 约定

项目中有两类模型规模，不应混为一谈：

- **教学锚点**：39M 用来打通端到端预训练，99M 用来学习超参数 sweep，213M 用来产出第一版正式 base model。
- **Scaling ladder 候选点**：用于拟合 scaling law、选择给定算力预算下的模型/数据组合，并验证对更大规模的预测。

当前 `v001` 暂时保留以下近似等比候选规模：

```text
p039m → p077m → p151m → p297m → p584m → p1150m
```

版本约定：

- `v001` 表示一个可比较的实验族；Tokenizer、语料配比、上下文长度、模型结构、优化器策略和参数统计口径应保持一致。
- `p039m` 等 ID 表示目标参数量，不是产品名称，也不代表当前配置已经精确达到该参数量。
- 一旦某个 ladder 开始产生正式实验结果，不再原地修改其比较口径；结构性变化进入 `v002`、`v003` 等新目录。
- 同一配置的重复实验通过独立 `run_id` 区分，禁止覆盖已有结果。
- 目前 [`configs/model/ladder/v001`](configs/model/ladder/v001) 中的 `p039m.yaml` 和 `p1150m.yaml` 都是 `dimensions_pending` 占位配置；其余四档尚未创建。

这里借鉴 Marin Delphi 的核心区分：**scaling law** 回答“给定算力预算应该训练多大的模型、使用多少 token”，**scaling heuristic/recipe** 回答“这个候选模型应该用什么 learning rate、batch size、optimizer 参数和 schedule 来训练”。不能先随意固定一组模型尺寸，再把它称为 compute-optimal scaling。

在冻结 `v001` 前，需要统一确定词表大小、是否共享输入/输出 Embedding、上下文长度、参数量统计口径、训练 token budget 和各档模型维度。39M/99M/213M 教学实验产生的数据会用于修订 ladder；因此当前六档名称是工程占位，不是最终科学结论。1.15B 训练应在小规模拟合、held-out 预测和双卡容量验证通过后再启动。

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

原始数据、处理中间产物、tokenized shards、checkpoint 和运行日志不提交到 Git；仓库只保存可复现处理过程所需的配置、脚本、manifest、统计摘要与数据来源说明。

## 实验记录规范

每个正式 run 至少记录以下信息：

- 身份信息：Git commit、配置快照、`run_id`、随机种子、环境与依赖版本。
- 训练曲线：train/validation loss、learning rate、token 数和累计 FLOPs。
- 数值稳定性：parameter norm、gradient norm、update norm、update/parameter ratio、LM Head norm、logZ 和 z-loss。
- 系统性能：tokens/s、step time、GPU 显存、TFLOPs 和 MFU。
- 数据信息：数据源配比、样本数量、过滤统计和训练 token 数。
- 产物信息：checkpoint、评测结果、profile 和 scaling report 的路径与校验信息。

MoE 不属于当前 dense miniLLaMA 的实现范围。如果后续单独开发 MoE，再增加 expert load、router entropy、router logits、dead experts 和 token drops 等指标，不与当前 dense ladder 混为同一实验族。

## 目录结构

```text
pretrain/
├── configs/          # 模型、训练、数据、评测和 scaling 配置
├── data/             # 本地数据阶段目录；大文件不进入 Git
├── scripts/          # 数据、训练、评测、推理、导出与 profiling 入口
├── src/dummym/       # 可复用 Python 包源码
├── tests/            # 单元、集成、数值与性能测试
├── runs/             # 每次运行的轻量元数据和索引
├── artifacts/        # checkpoint、Tokenizer、评测及 profile 产物目录
├── reports/          # scaling、消融实验和生成报告
├── docs/             # 设计文档与开发说明
├── notebooks/        # 探索性分析；正式逻辑应迁移到 src/ 或 scripts/
└── pyproject.toml    # Python 包、依赖与测试配置
```

## 学习路线与里程碑

里程碑是按依赖关系排列的课程，不以“代码写完”作为唯一完成标准。每一阶段都必须留下可复现配置、测试、run card、指标和简短复盘，才能进入下一阶段。

| Milestone | 要真正学会什么 | 主要交付物与通过条件 |
| --- | --- | --- |
| **M0 · Foundations** | 自己实现 Transformer 与 BPE Tokenizer | 完成 RMSNorm、RoPE、GQA、SwiGLU、causal loss、采样生成；Tokenizer 可训练、保存、重载并稳定复现；通过 shape、mask、数值测试和 tiny-corpus overfit。当前阶段进行中，Qwen Tokenizer 只用于临时生成冒烟测试。 |
| **M1 · 39M from scratch** | 第一次完整预训练，而不是只会调用 Trainer | 建立数据 manifest、清洗、去重、tokenization、packing、train/validation split；完成单卡训练、定期验证、checkpoint 保存与恢复；loss 明显下降且恢复训练轨迹合理一致。 |
| **M2 · 99M recipe sweep** | 学会控制变量和选择训练 recipe | 在固定模型、数据、token budget 和随机种子策略下，对 LR、AdamW betas/weight decay、warmup/decay schedule 做 sweep；按预先声明的验证 loss、稳定性与吞吐指标选择 recipe，并保留失败实验。新型优化器研究留到 M6，避免变量混杂。 |
| **M3 · Distributed systems** | DDP、TorchTitan/FSDP2 与 profiling | 对齐单卡和双卡的首步/短程 loss；验证梯度累积、混合精度、分布式 checkpoint 与恢复；用 `torch.profiler`/Nsight 分析吞吐、显存和通信瓶颈。此阶段先做 dense data parallel，EP 留到 M7。 |
| **M4 · 213M base pretraining** | 运行第一版“正式”base model 训练 | 冻结 Tokenizer、数据 recipe 和 dense 训练 recipe；在两张 H20 上完成可恢复训练；产出 checkpoint、训练报告、base eval、模型卡和完整 provenance。 |
| **M5 · Mini-Delphi scaling** | IsoFLOP、scaling law、scaling recipe 与外推验证 | 设计多个 compute budget 和候选 `(参数量, token 数)`；小规模点使用重复 seed；拟合并报告不确定性；用 held-out 规模检验预测，再决定是否运行最高至 1.15B 的 ladder。`Mini-Delphi` 是 DummyM 的教学实验名，不代表 Marin 官方 Delphi 的复现结果。 |
| **M6 · Optimizer research** | Muon 与 Hyperball 系列方法如何公平比较 | 以 M2/M5 的 AdamW recipe 为固定基线，对 Muon 及 Hyperball 约束版本（如 AdamH/MuonH）做单变量 A/B；记录 loss、吞吐、参数范数和 update/parameter ratio；新优化器使用独立 scaling heuristic，不能直接沿用 AdamW 最优参数。 |
| **M7 · Mini-MoE systems** | Sparse MoE、Router、负载均衡和 Expert Parallel | 先实现可测试的 top-k router 与专家层，再加入容量、token dispatch/combine、负载与丢 token 指标；将 Quantile Balancing（QB）作为独立路由实验；与 active-parameter/compute 匹配的 dense baseline 比较，最后接入 EP 并 profile 通信。 |
| **M8 · Midtraining & cooldown** | 数据混合变化与学习率退火如何影响能力 | 从同一 base checkpoint 分叉，对高质量/领域数据配比、阶段 token budget 和 cooldown schedule 做受控实验；同时看通用能力保持、目标能力增益和遗忘，而不只看单项 benchmark。 |
| **M9 · Post-training** | SFT、偏好优化与可验证奖励强化学习 | 定义 chat template 和数据审计；完成 SFT、DPO、GRPO/RLVR 的目标函数、reference policy/奖励与评测。学习顺序可按 SFT → DPO → GRPO，但默认应从同一 SFT checkpoint 建立 DPO 与 GRPO 对照分支，不假设三者必须串行才正确。 |

### 每个里程碑的统一完成标准

一个 milestone 标记完成前，应至少满足：

1. **正确性**：有单元/集成测试、短程 loss 检查，并能解释关键 tensor shape 和数学目标。
2. **可恢复性**：中断后能从 checkpoint 恢复模型、优化器、scheduler、数据位置和随机状态。
3. **可复现性**：保存 commit、完整配置、依赖与硬件信息、数据/Tokenizer 标识和 seed。
4. **可比较性**：实验只改变声明的变量，基线、预算、评测协议和统计口径一致。
5. **可观察性**：同时记录质量、数值稳定性、吞吐、显存和失败原因。
6. **可交付性**：产出 run card、指标文件、checkpoint/报告索引，以及“学到了什么”的复盘。

## 参考项目与方法

- [Marin](https://github.com/marin-community/marin)：开放记录数据、训练、评测、实验决策与失败结果的方法论参考。
- [Marin: Train a language model](https://marin.readthedocs.io/en/latest/tutorials/train-an-lm/)：从数据混合、模型配置到训练任务的完整示例。
- [Marin scaling heuristic recipe](https://github.com/marin-community/marin/blob/main/docs/recipes/add_scaling_heuristic.md)：区分 scaling law 与 training heuristic，并组织 IsoFLOP sweep。
- [Delphi scaling suite](https://github.com/marin-community/marin/issues/1337)：开放 scaling suite、重复 seed、统一评测和可复现数据顺序的参考。

这些资料用于学习实验设计，不意味着 DummyM 的结果可以直接与 Marin/Delphi 对比。硬件、框架、Tokenizer、数据、计算预算和参数统计口径不同，任何数值比较都必须先重新建立公平基线。
