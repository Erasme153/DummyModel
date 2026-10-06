# Experiments

这是学习项目，实验记录保持简单：一个实验目录只放一个 `README.md`。
`expNNN` 在各里程碑目录内分别编号，所以不同里程碑可以都有 `exp001`。

```text
experiments/
├── m00_foundations/
│   └── exp001_tiny_overfit/
│       └── README.md
├── m01_pretraining/
│   └── exp001_39m/
│       └── README.md
├── m02_recipe/
│   ├── exp001_lr_sweep/
│   │   └── README.md
│   └── exp002_warmup/
│       └── README.md
├── m03_distributed/
│   └── exp001_parallelism/
│       └── README.md
├── m04_base/
│   └── exp001_213m/
│       └── README.md
└── m05_scaling/
    └── exp001_isoflop/
        └── README.md
```

最新报告：[M5：等算力对照与留出预算验证](m05_scaling/exp001_isoflop/README.md)。
三档近似等算力预算中，39M / 547M、39M / 856M、99M / 1B tokens 的
已测最低同集验证 loss 分别为 3.272709、3.209490、3.010837。
213M base model 的独立评测见
[M4 报告](m04_base/exp001_213m/README.md)。

每份 README 只需要回答五个问题：

1. 为什么做？
2. 用什么数据和关键参数？
3. 执行什么命令？
4. 得到了什么结果？
5. 结论和下一步是什么？

默认不创建实验 YAML、独立 results 文件、run ID 或配置快照。只有当参数多到
命令行难以维护，或者需要批量对比实验时，才增加可复用配置文件。

训练自动生成的 checkpoint、TensorBoard 日志和 summary 放在 `runs/`，不提交
Git；实验 README 只记录复现命令、关键指标和产物路径。
