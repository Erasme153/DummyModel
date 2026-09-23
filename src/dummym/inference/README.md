# Inference

## 已训练 checkpoint

`scripts/inference/pretrained_checkpoint_demo.py` 是通用的 checkpoint 生成入口。
它只要求 checkpoint 包含 `model_config`、`model_state_dict`，并能取得训练时的
`tokenizer.json`，因此可以用于 M0、M1 和后续保存为完整单文件 `.pt` 的模型。

从项目根目录运行：

```bash
python scripts/inference/pretrained_checkpoint_demo.py \
  --checkpoint runs/m01_39m/checkpoint.pt \
  --device cuda:0 \
  --prompt "The future of artificial intelligence" \
  --max-new-tokens 32
```

默认 `temperature=0`，即贪心生成。需要采样时显式设置，例如：

```bash
python scripts/inference/pretrained_checkpoint_demo.py \
  --checkpoint runs/m01_39m/checkpoint.pt \
  --device cuda:0 \
  --prompt "The future of artificial intelligence" \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --seed 2026
```

checkpoint 从其他机器复制过来后，其中记录的 tokenizer 绝对路径可能失效；此时
用 `--tokenizer /current/path/tokenizer.json` 覆盖。当前入口读取的是完整单文件
checkpoint，未来的 FSDP 分片 checkpoint 需要先合并或使用相应的分布式加载器。

`tiny_checkpoint_demo.py` 仍是 M0 专用验收脚本：除了生成，它还会用原始 tiny
语料重新计算 loss 和 next-token accuracy。不要用它加载 M1 checkpoint。

## 随机权重冒烟测试

`generation.py` contains a deliberately simple autoregressive loop. It
recomputes the full prompt at every step and is intended for correctness tests,
not serving performance.

Run the random-weight end-to-end demo from the project root:

```bash
conda activate dummym
python scripts/inference/random_prompt_demo.py \
  "你好，请介绍一下你自己。" \
  --tokenizer /path/to/tokenizer.json \
  --max-new-tokens 24 \
  --seed 2026
```

Only the tokenizer is loaded; model code and weights from that model are not
loaded. Since the miniLLaMA weights are random, generated text is expected to be
meaningless. The purpose is to validate prompt encoding, model forward,
autoregressive sampling, and decoding as one complete pipeline.
