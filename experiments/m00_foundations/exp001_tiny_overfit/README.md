# M0：Tiny Corpus Overfit

## 目的

验证 Tokenizer、定长 packing、causal LM loss、反向传播、AdamW、
TensorBoard 和 checkpoint 保存/加载能够连通。这个实验只检查模型能否记住
固定 batch，不衡量泛化能力。

## 运行

```bash
conda activate dummym
python scripts/train/tiny_overfit.py \
  --tokenizer /path/to/Mistral-7B-v0.1/tokenizer.json \
  --device cuda
```

关键设置：16 行语料、Mistral 32K Tokenizer、每篇文档追加 EOS、不添加 BOS、
`4 × 128` 固定 batch、2,136,384 参数、AdamW、恒定学习率 `1e-3`、seed 2026。

## 结果

| 指标 | 结果 |
| --- | ---: |
| 初始 loss | 10.381273 |
| 最终 loss | 0.049335 |
| checkpoint 重新加载后的 loss | 0.047793 |
| next-token accuracy | 1.0000 |
| 完成步数 | 223 |

checkpoint 严格加载时所有参数名称均匹配。用训练语料前缀
`The small language model` 做 greedy decoding，模型生成：

```text
learns to predict the next token from the tokens that came before it.</s>
```

测试命令：

```bash
python scripts/inference/tiny_checkpoint_demo.py --device cuda
```

TensorBoard：

```bash
tensorboard --logdir runs/m00_tiny_overfit/tensorboard
```

## 结论

**通过。** 当前最小训练链路能够完全拟合固定训练数据，保存后的 checkpoint
也能恢复并复现训练内容。这个结果不能说明模型具备通用语言能力。

下一步是确定正式的 BOS/EOS 约定，并验证从 checkpoint 继续训练是否正常。

相关文件：

- 语料：`data/tiny_corpus.txt`
- 训练：`scripts/train/tiny_overfit.py`
- checkpoint 测试：`scripts/inference/tiny_checkpoint_demo.py`
- 本地产物：`runs/m00_tiny_overfit/`
