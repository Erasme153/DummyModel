from dataclasses import replace

import pytest
import torch

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from dummym.models.llama_like.moe import TopKMoE, qb_dual_update


def config(**overrides):
    base = MiniLlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=8)
    return replace(base, **overrides)


def test_dense_config_keeps_old_checkpoint_contract():
    dense = config()
    assert "num_experts" not in dense.to_dict()
    assert MiniLlamaConfig.from_dict(dense.to_dict()) == dense
    with pytest.raises(ValueError, match="num_experts"):
        config(num_experts=1)
    old_moe = config(num_experts=4, experts_per_token=2, expert_intermediate_size=16)
    assert "moe_routing" not in old_moe.to_dict()
    assert MiniLlamaConfig.from_dict(old_moe.to_dict()) == old_moe
    with pytest.raises(ValueError, match="QB requires"):
        config(num_experts=2, experts_per_token=2, moe_routing="qb")


def test_qb_dual_update_balances_skewed_logits():
    torch.manual_seed(4)
    logits = torch.randn(256, 4)
    logits[:, 0] += 1.2
    beta = torch.zeros(4)
    initial, _ = qb_dual_update(logits, 2, beta)
    assert torch.bincount(initial.flatten(), minlength=4)[0] > 200
    for _ in range(4):
        _, beta = qb_dual_update(logits, 2, beta)
    selected, _ = qb_dual_update(logits, 2, beta)
    counts = torch.bincount(selected.flatten(), minlength=4)
    assert (counts - 128).abs().max().item() <= 2


def test_qb_uses_bias_for_selection_but_original_scores_for_gates():
    torch.manual_seed(16)
    moe = TopKMoE(config(num_experts=4, experts_per_token=2,
                         expert_intermediate_size=16, moe_routing="qb"))
    inputs = torch.randn(2, 4, 16)
    with torch.no_grad():
        moe.qb_beta.copy_(torch.tensor([0.5, 0.0, 0.0, -0.5]))
    logits = moe.router(inputs.reshape(-1, 16)).float()
    selected = (logits - moe.qb_beta).topk(2, dim=-1).indices
    output = moe(inputs)
    torch.testing.assert_close(output.selected_counts,
                               torch.bincount(selected.flatten(), minlength=4))
    flat = inputs.reshape(-1, 16)
    probabilities = logits.softmax(dim=-1).gather(1, selected)
    gates = probabilities / probabilities.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(flat)
    for expert_id, expert in enumerate(moe.experts):
        values = expert(flat)
        expected += values * (gates * (selected == expert_id)).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(output.hidden_states, expected.reshape_as(inputs), atol=1e-6, rtol=1e-5)
    assert output.qb_beta_candidate is not None
    output.hidden_states.square().mean().backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum().item() > 0
    before = moe.qb_beta.clone()
    moe.eval()
    with torch.no_grad():
        evaluated = moe(inputs)
    assert evaluated.qb_beta_candidate is None
    torch.testing.assert_close(moe.qb_beta, before, atol=0, rtol=0)
    assert "qb_beta" in moe.state_dict()


def test_topk_routing_combines_experts_and_backpropagates():
    torch.manual_seed(12)
    moe = TopKMoE(config(num_experts=2, experts_per_token=2, expert_intermediate_size=16))
    inputs = torch.randn(2, 4, 16, requires_grad=True)
    output = moe(inputs)
    flat = inputs.reshape(-1, 16)
    probabilities = moe.router(flat).float().softmax(dim=-1)
    expected = (probabilities[:, :1] * moe.experts[0](flat).float() +
                probabilities[:, 1:] * moe.experts[1](flat).float()).reshape_as(inputs)
    torch.testing.assert_close(output.hidden_states, expected, atol=1e-6, rtol=1e-5)
    assert output.selected_counts.tolist() == [8, 8]
    assert output.kept_counts.tolist() == [8, 8]
    assert output.dropped_tokens.item() == 0
    (output.hidden_states.square().mean() + 0.01 * output.aux_loss).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in moe.parameters())


def test_capacity_drops_assignments_without_nan_or_unused_experts():
    torch.manual_seed(13)
    moe = TopKMoE(config(num_experts=4, experts_per_token=2,
                         expert_intermediate_size=16, capacity_factor=0.25))
    output = moe(torch.randn(2, 8, 16))
    assert output.selected_counts.sum().item() == 32
    assert output.kept_counts.sum().item() == 8
    assert output.dropped_tokens.item() > 0
    assert torch.isfinite(output.hidden_states).all()
    (output.hidden_states.square().mean() + 0.01 * output.aux_loss).backward()
    assert all(parameter.grad is not None for parameter in moe.parameters())


def test_top1_task_loss_trains_router_without_auxiliary_loss():
    torch.manual_seed(15)
    moe = TopKMoE(config(num_experts=2, experts_per_token=1, expert_intermediate_size=16))
    output = moe(torch.randn(2, 8, 16))
    output.hidden_states.square().mean().backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum().item() > 0


def test_moe_lm_keeps_cross_entropy_separate_from_router_loss():
    torch.manual_seed(14)
    model = MiniLlamaForCausalLM(config(num_experts=4, experts_per_token=2,
                                         expert_intermediate_size=16))
    tokens = torch.randint(0, 64, (2, 8))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 8, 64)
    assert output.loss is not None and output.aux_loss is not None
    assert output.router_stats["selected_counts"].shape == (2, 4)
    assert output.router_stats["selected_counts"].sum().item() == 2 * 2 * 8 * 2
    assert output.router_stats["dropped_tokens"].sum().item() == 0
