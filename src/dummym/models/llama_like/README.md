# miniLLaMA reference model

This package implements the single-device model mathematics only:

```text
token embedding
  -> N x (RMSNorm -> GQA causal SDPA -> residual
           -> RMSNorm -> SwiGLU MLP -> residual)
  -> RMSNorm
  -> bias-free LM head
  -> vocabulary logits
```

RoPE is applied to query and key states before attention. GQA projects fewer
key/value heads than query heads, then shares each key/value head across its
query-head group. Embedding and LM-head weights can be tied.

## Minimal use

```python
import torch

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM

config = MiniLlamaConfig(
    vocab_size=32_000,
    hidden_size=256,
    intermediate_size=688,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=2_048,
)
model = MiniLlamaForCausalLM(config)
input_ids = torch.randint(0, config.vocab_size, (2, 128))
output = model(input_ids, labels=input_ids)
output.loss.backward()
```

This reference version intentionally does not include KV caching, generation,
FSDP2, tensor parallelism, activation checkpointing, or Hugging Face checkpoint
conversion. Those features belong in separate integration layers.
