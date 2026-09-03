from __future__ import annotations

import pytest
import torch

from dummym.inference import generate_token_ids
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM


def test_generation_appends_requested_number_of_tokens() -> None:
    torch.manual_seed(17)
    config = MiniLlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=80,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    model = MiniLlamaForCausalLM(config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generator = torch.Generator().manual_seed(19)

    generated = generate_token_ids(
        model,
        prompt,
        max_new_tokens=5,
        temperature=0.8,
        top_k=8,
        generator=generator,
    )

    assert generated.shape == (1, 8)
    torch.testing.assert_close(generated[:, :3], prompt)
    assert generated.min() >= 0
    assert generated.max() < config.vocab_size


def test_generation_respects_context_limit() -> None:
    config = MiniLlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=5,
    )
    model = MiniLlamaForCausalLM(config)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    generated = generate_token_ids(
        model,
        prompt,
        max_new_tokens=10,
        temperature=0,
    )

    assert generated.shape == (1, 5)


def test_top_p_can_reduce_sampling_to_the_highest_probability_token() -> None:
    torch.manual_seed(23)
    config = MiniLlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )
    model = MiniLlamaForCausalLM(config).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        expected = model(prompt).logits[:, -1, :].argmax(dim=-1)

    generated = generate_token_ids(
        model,
        prompt,
        max_new_tokens=1,
        temperature=1.0,
        top_k=None,
        top_p=1e-6,
        generator=torch.Generator().manual_seed(29),
    )

    torch.testing.assert_close(generated[:, -1], expected)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1])
def test_generation_rejects_invalid_top_p(top_p: float) -> None:
    config = MiniLlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = MiniLlamaForCausalLM(config)
    prompt = torch.tensor([[1]], dtype=torch.long)

    with pytest.raises(ValueError, match="top_p"):
        generate_token_ids(
            model,
            prompt,
            max_new_tokens=1,
            top_p=top_p,
        )
