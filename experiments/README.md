# Experiments

这是学习项目，实验记录保持简单：一个实验目录只放一个 `README.md`。

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
└── m03_distributed/
    └── exp001_ddp/
        └── README.md
```

最新报告：[M3：单卡 / 双卡 DDP](m03_distributed/exp001_ddp/README.md)。
99M 的双 GPU 完整预算对照已完成：最终验证 loss 单卡 3.657353、双卡 3.657155，
稳态全局训练吞吐双卡约为单卡的 1.95 倍。40 步 Nsight profiling 已完成，
FSDP2 尚未实现。前序
[M2 Warmup 对比](m02_recipe/exp002_warmup/README.md)
确定的基线仍为 LR=1e-3、warmup=300。

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
