"""Grouped-query causal self-attention backed by PyTorch SDPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniLlamaConfig
from .rope import RotaryEmbedding, apply_rotary_position_embeddings


def repeat_key_value_heads(
    hidden_states: torch.Tensor,
    num_repeats: int,
) -> torch.Tensor:
    """Expand shared key/value heads to the number of query heads.

    Input and output layouts are `[batch, heads, sequence, head_dim]`.
    """

    if num_repeats == 1:
        return hidden_states
    batch_size, num_kv_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_kv_heads,
        num_repeats,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(
        batch_size,
        num_kv_heads * num_repeats,
        sequence_length,
        head_dim,
    )


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with fewer key/value heads than query heads."""

    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_embedding = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
        )

    def _shape(
        self,
        hidden_states: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        return hidden_states.view(
            batch_size,
            sequence_length,
            num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = self._shape(
            self.q_proj(hidden_states), self.num_attention_heads
        )
        key = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        cos, sin = self.rotary_embedding(position_ids, dtype=query.dtype)
        query, key = apply_rotary_position_embeddings(query, key, cos, sin)

        key = repeat_key_value_heads(key, self.num_key_value_groups)
        value = repeat_key_value_heads(value, self.num_key_value_groups)

        sdpa_mask: torch.Tensor | None = None
        is_causal = attention_mask is None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask must have shape [batch_size, sequence_length]"
                )
            causal_mask = torch.ones(
                (sequence_length, sequence_length),
                device=hidden_states.device,
                dtype=torch.bool,
            ).tril()
            key_padding_mask = attention_mask.to(dtype=torch.bool)[:, None, None, :]
            sdpa_mask = causal_mask[None, None, :, :] & key_padding_mask

        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=sdpa_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        return self.o_proj(attention_output)
