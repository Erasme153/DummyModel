"""Minimal autoregressive generation for correctness smoke tests.

This implementation intentionally recomputes the full prefix on every step. It
is easy to inspect and sufficient for validating a random model end to end. A
production inference path should add a KV cache or use vLLM instead.
"""

from __future__ import annotations

import torch

from dummym.models.llama_like import MiniLlamaForCausalLM


@torch.inference_mode()
def generate_token_ids(
    model: MiniLlamaForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = 50,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append autoregressively sampled token IDs to one prompt.

    Args:
        model: A miniLLaMA causal language model.
        input_ids: Prompt token IDs with shape ``[1, prompt_length]``.
        max_new_tokens: Maximum number of tokens to append.
        temperature: Sampling temperature. Set to 0 for greedy decoding.
        top_k: Restrict sampling to the highest-scoring K tokens. ``None`` uses
            the full vocabulary.
        eos_token_id: Stop after this token is sampled, if provided.
        generator: Optional device-local random generator for reproducibility.

    Returns:
        Prompt and generated IDs together, with shape
        ``[1, prompt_length + generated_length]``.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one prompt token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")

    model.eval()
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        if generated_ids.shape[1] >= model.config.max_position_embeddings:
            break

        # Without a KV cache, every step recomputes the complete prefix.
        next_token_logits = model(generated_ids).logits[:, -1, :]

        if temperature == 0:
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits.float() / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.shape[-1])
                top_values, top_indices = torch.topk(next_token_logits, k=k, dim=-1)
                probabilities = torch.softmax(top_values, dim=-1)
                sampled_top_index = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                next_token_id = top_indices.gather(-1, sampled_top_index)
            else:
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )

        generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    return generated_ids
