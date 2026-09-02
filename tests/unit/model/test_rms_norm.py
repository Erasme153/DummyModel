from __future__ import annotations

import torch

from dummym.models.llama_like.norm import RMSNorm


def test_rms_norm_matches_reference() -> None:
    torch.manual_seed(3)
    hidden_states = torch.randn(2, 5, 16)
    norm = RMSNorm(16, eps=1e-5)
    norm.weight.data.uniform_(0.5, 1.5)

    actual = norm(hidden_states)
    expected = hidden_states * torch.rsqrt(
        hidden_states.square().mean(dim=-1, keepdim=True) + 1e-5
    )
    expected = expected * norm.weight

    torch.testing.assert_close(actual, expected)
