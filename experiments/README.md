# Experiments

此目录保存可提交到 Git 的实验定义与结论，不保存训练产生的大文件。模型、数据和训练组件的可复用配置位于 `configs/`；本地运行日志位于 `runs/`；checkpoint 和 profile 位于 `artifacts/`。

## 推荐结构

按 milestone 和实验编号组织，在真正开始某阶段时再创建对应目录：

```text
experiments/
├── README.md
├── templates/
│   └── experiment.yaml
├── m00_foundations/
│   └── exp001_tiny_overfit/
│       ├── experiment.yaml
│       ├── README.md
│       └── results.md
├── m01_39m_pretrain/
├── m02_99m_sweeps/
└── m05_mini_delphi/
```

目录名使用 `expNNN_short_name`。同一个实验中的不同 seed、学习率或重复运行使用不同 `run_id`，不新建 Git 分支或复制实验目录。

## 生命周期

1. 从 `templates/experiment.yaml` 复制实验卡，填写假设、基线、唯一变量、预算、指标和停止条件。
2. 在短期 `exp/<name>` 分支中提交实验卡和所需配置，测试后合并到 `main`。
3. 从干净的 `main` commit 启动训练，将 resolved config、环境和日志写入 `runs/`。
4. 将 W&B/TensorBoard run ID、artifact URI、哈希、状态和摘要回填到实验卡。
5. 在 `results.md` 记录成功或失败的结论；重要结果再整理到 `reports/` 并创建 milestone tag。

实验卡一旦关联正式 run，不再改写其原始假设、变量和验收标准。需要改变实验设计时新建 `expNNN`；允许追加 run 索引、结果和结论。

## 提交与忽略规则

提交到 Git：

- `experiment.yaml`
- 人工编写的 `README.md`、`results.md`
- 小型汇总表、图表源数据和外部 artifact 索引

不提交到 Git：

- checkpoint、优化器状态和数据 shard
- stdout、逐 step 原始日志和 profiler trace
- W&B 本地缓存及任何密钥

