"""Configuration for the reference miniLLaMA implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(slots=True)
class MiniLlamaConfig:
    """Architecture-only configuration for a decoder-only Transformer.

    Training and distributed settings deliberately do not live here. This keeps
    the model definition usable on one device before TorchTitan/FSDP2 is applied.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 2048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    num_experts: int = 0
    experts_per_token: int = 2
    expert_intermediate_size: int | None = None
    capacity_factor: float | None = None
    moe_routing: str = "topk"

    def __post_init__(self) -> None:
        positive_int_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        )
        for field_name in positive_int_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if self.num_experts < 0 or (self.num_experts and self.experts_per_token > self.num_experts):
            raise ValueError("num_experts must be >= experts_per_token, or zero for dense")
        if self.experts_per_token <= 0:
            raise ValueError("experts_per_token must be positive")
        if self.expert_intermediate_size is not None and self.expert_intermediate_size <= 0:
            raise ValueError("expert_intermediate_size must be positive")
        if self.capacity_factor is not None and (
            not self.num_experts or not math.isfinite(self.capacity_factor) or self.capacity_factor <= 0
        ):
            raise ValueError("capacity_factor requires MoE and must be finite and positive")
        if self.moe_routing not in ("topk", "qb"):
            raise ValueError("moe_routing must be topk or qb")
        if self.moe_routing == "qb" and (not self.num_experts or self.experts_per_token >= self.num_experts):
            raise ValueError("QB requires 0 < experts_per_token < num_experts")
        if self.moe_routing == "qb" and self.capacity_factor is not None:
            raise ValueError("QB currently requires no capacity limit")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        # Preserve the v1 dense checkpoint contract when MoE is disabled.
        if not self.num_experts:
            for name in ("num_experts", "experts_per_token", "expert_intermediate_size", "capacity_factor",
                         "moe_routing"):
                values.pop(name)
        elif self.moe_routing == "topk":
            # Older MoE checkpoint contracts did not include this default.
            values.pop("moe_routing")
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MiniLlamaConfig":
        return cls(**values)
