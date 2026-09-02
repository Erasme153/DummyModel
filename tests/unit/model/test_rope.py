from __future__ import annotations

import torch

from dummym.models.llama_like.rope import (
    RotaryEmbedding,
    apply_rotary_position_embeddings,
)


def test_rope_preserves_vector_norm() -> None:
    torch.manual_seed(5)
    query = torch.randn(2, 4, 7, 8)
    key = torch.randn(2, 2, 7, 8)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)
    rope = RotaryEmbedding(head_dim=8)
    cos, sin = rope(position_ids, dtype=query.dtype)

    rotated_query, rotated_key = apply_rotary_position_embeddings(
        query, key, cos, sin
    )

    torch.testing.assert_close(
        rotated_query.norm(dim=-1), query.norm(dim=-1), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        rotated_key.norm(dim=-1), key.norm(dim=-1), rtol=1e-5, atol=1e-6
    )
