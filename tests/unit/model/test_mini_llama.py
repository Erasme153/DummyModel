from __future__ import annotations

import pytest
import torch

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM


@pytest.fixture
def tiny_config() -> MiniLlamaConfig:
    return MiniLlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=176,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


def test_forward_shape_and_tied_embeddings(tiny_config: MiniLlamaConfig) -> None:
    model = MiniLlamaForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (3, 17))

    output = model(input_ids)

    assert output.logits.shape == (3, 17, tiny_config.vocab_size)
    assert output.loss is None
    assert model.lm_head.weight is model.embed_tokens.weight


def test_gqa_uses_fewer_key_value_projections(
    tiny_config: MiniLlamaConfig,
) -> None:
    attention = MiniLlamaForCausalLM(tiny_config).layers[0].self_attn

    assert attention.q_proj.out_features == 8 * tiny_config.head_dim
    assert attention.k_proj.out_features == 2 * tiny_config.head_dim
    assert attention.v_proj.out_features == 2 * tiny_config.head_dim
    assert attention.num_key_value_groups == 4


def test_causal_attention_cannot_see_future_tokens(
    tiny_config: MiniLlamaConfig,
) -> None:
    torch.manual_seed(7)
    model = MiniLlamaForCausalLM(tiny_config).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed_future = torch.tensor([[1, 2, 3, 91, 92, 93]])

    with torch.no_grad():
        original_logits = model(original).logits
        changed_logits = model(changed_future).logits

    torch.testing.assert_close(
        original_logits[:, :3],
        changed_logits[:, :3],
        rtol=1e-5,
        atol=1e-6,
    )


def test_causal_lm_loss_backpropagates(tiny_config: MiniLlamaConfig) -> None:
    torch.manual_seed(11)
    model = MiniLlamaForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 12))

    output = model(input_ids, labels=input_ids)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_invalid_gqa_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_attention_heads"):
        MiniLlamaConfig(
            vocab_size=128,
            hidden_size=60,
            intermediate_size=160,
            num_hidden_layers=2,
            num_attention_heads=6,
            num_key_value_heads=4,
        )


def test_maximum_sequence_length_is_enforced(
    tiny_config: MiniLlamaConfig,
) -> None:
    model = MiniLlamaForCausalLM(tiny_config)
    input_ids = torch.zeros(
        (1, tiny_config.max_position_embeddings + 1),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(input_ids)
