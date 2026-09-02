# Inference smoke test

`generation.py` contains a deliberately simple autoregressive loop. It
recomputes the full prompt at every step and is intended for correctness tests,
not serving performance.

Run the random-weight end-to-end demo from the project root:

```bash
/diff/workspace/wyl/openpi/.venv/bin/python \
  scripts/inference/random_prompt_demo.py \
  "你好，请介绍一下你自己。" \
  --max-new-tokens 24 \
  --seed 2026
```

The default tokenizer is borrowed read-only from:

```text
/diff/models/Qwen/Qwen3.5-2B/tokenizer.json
```

Only the tokenizer is loaded. Qwen model code and weights are not loaded. Since
the miniLLaMA weights are random, generated text is expected to be meaningless.
The purpose is to validate prompt encoding, model forward, autoregressive
sampling, and decoding as one complete pipeline.
