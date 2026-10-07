"""Top-k SwiGLU experts, with optional two-rank expert parallel dispatch."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single as differentiable_all_to_all_single
from torch import nn

from .config import MiniLlamaConfig
from .mlp import SwiGLUMLP


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    selected_counts: torch.Tensor
    kept_counts: torch.Tensor
    dropped_tokens: torch.Tensor
    router_entropy: torch.Tensor
    qb_beta_candidate: torch.Tensor | None = None


@torch.no_grad()
def qb_dual_update(scores: torch.Tensor, top_k: int, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Route with the previous bias and estimate the next per-expert quantile.

    The (k+1)-th adjusted score is each token's cutoff. Each column's
    target-load quantile of the unadjusted score minus that cutoff gives the
    next dual bias. Selection and the candidate deliberately use old beta.
    """
    tokens, experts = scores.shape
    if not 0 < top_k < experts:
        raise ValueError("QB requires 0 < top_k < num_experts")
    adjusted = (scores.float() - beta.float()).topk(top_k + 1, dim=-1)
    indices = adjusted.indices[:, :top_k]
    if tokens < 2:
        return indices, beta.detach().float().clone()
    cutoff = adjusted.values[:, -1:]
    target = max(1, min(tokens - 1, math.ceil(tokens * top_k / experts)))
    candidate = (scores.float() - cutoff).topk(target + 1, dim=0).values[-1]
    return indices, candidate


class TopKMoE(nn.Module):
    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()
        if not config.num_experts:
            raise ValueError("TopKMoE requires num_experts > 0")
        self.num_experts = config.num_experts
        self.top_k = config.experts_per_token
        self.capacity_factor = config.capacity_factor
        self.routing = config.moe_routing
        if self.routing == "qb":
            self.register_buffer("qb_beta", torch.zeros(config.num_experts, dtype=torch.float32))
        width = config.expert_intermediate_size or config.intermediate_size
        expert_config = replace(config, intermediate_size=width)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUMLP(expert_config) for _ in range(config.num_experts))
        self.ep_rank = None
        self.ep_group = None

    def shard_experts(self, rank: int, world_size: int, group=None) -> None:
        """Keep global expert IDs belonging to this rank after full initialization."""
        if self.ep_rank is not None or world_size != 2 or self.num_experts % world_size:
            raise ValueError("EP requires two ranks and a divisible number of experts")
        if not dist.is_initialized() or dist.get_world_size(group) != world_size:
            raise ValueError("EP requires an initialized two-rank process group")
        if not 0 <= rank < world_size or self.capacity_factor is not None or self.routing != "topk":
            raise ValueError("EP currently supports top-k routing without capacity limits")
        per_rank = self.num_experts // world_size
        start = rank * per_rank
        self.experts = nn.ModuleDict((str(i), self.experts[i]) for i in range(start, start + per_rank))
        self.ep_rank = rank
        self.ep_group = group

    def _dispatch_experts(self, flat: torch.Tensor, indices: torch.Tensor,
                          gates: torch.Tensor) -> torch.Tensor:
        """Send selected tokens to expert owners and return outputs in source order."""
        count, width = flat.shape
        per_rank = self.num_experts // 2
        expert_ids = indices.reshape(-1)
        destinations = torch.div(expert_ids, per_rank, rounding_mode="floor")
        order = destinations.argsort(stable=True)
        send_counts = torch.bincount(destinations, minlength=2)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        send_splits, recv_splits = send_counts.tolist(), recv_counts.tolist()

        token_ids = torch.arange(count, device=flat.device).repeat_interleave(self.top_k)[order]
        send_hidden = flat.index_select(0, token_ids).contiguous()
        send_expert_ids = expert_ids.index_select(0, order).contiguous()
        total_received = sum(recv_splits)
        received = differentiable_all_to_all_single(
            torch.empty((total_received, width), dtype=flat.dtype, device=flat.device), send_hidden,
            output_split_sizes=recv_splits, input_split_sizes=send_splits, group=self.ep_group)
        received_expert_ids = torch.empty(total_received, dtype=torch.long, device=flat.device)
        dist.all_to_all_single(received_expert_ids, send_expert_ids,
                               output_split_sizes=recv_splits, input_split_sizes=send_splits,
                               group=self.ep_group)

        processed = torch.zeros((total_received, width), dtype=torch.float32, device=flat.device)
        for global_id, expert in self.experts.items():
            positions = (received_expert_ids == int(global_id)).nonzero(as_tuple=True)[0]
            # The empty call keeps zero-token experts in the backward graph.
            values = expert(received.index_select(0, positions)).float()
            processed = processed.index_add(0, positions, values)
        returned = differentiable_all_to_all_single(
            torch.empty((len(order), width), dtype=torch.float32, device=flat.device), processed,
            output_split_sizes=send_splits, input_split_sizes=recv_splits, group=self.ep_group)
        combined = torch.zeros((count, width), dtype=torch.float32, device=flat.device)
        combined = combined.index_add(0, token_ids, returned * gates.reshape(-1)[order, None])
        if self.top_k > 1:
            gate_sum = torch.zeros(count, dtype=torch.float32, device=flat.device)
            gate_sum = gate_sum.index_add(0, token_ids, gates.reshape(-1)[order])
            combined = combined / gate_sum.clamp_min(1e-12)[:, None]
        return combined

    def forward(self, hidden_states: torch.Tensor) -> MoEOutput:
        batch, length, width = hidden_states.shape
        flat = hidden_states.reshape(-1, width)
        count = flat.shape[0]
        # Router probabilities and balancing statistics stay FP32 under BF16 autocast.
        logits = self.router(flat).float()
        probabilities = logits.softmax(dim=-1)
        qb_beta_candidate = None
        if self.routing == "qb":
            if self.training and torch.is_grad_enabled():
                indices, qb_beta_candidate = qb_dual_update(logits.detach(), self.top_k, self.qb_beta)
            else:
                indices = (logits.detach() - self.qb_beta).topk(self.top_k, dim=-1).indices
            selected_probabilities = probabilities.gather(1, indices)
        else:
            selected_probabilities, indices = probabilities.topk(self.top_k, dim=-1)
        # Top-1 keeps its router probability so the task loss trains the router;
        # top-k>1 normalizes the selected probabilities to sum to one.
        gates = (selected_probabilities if self.top_k == 1 else
                 selected_probabilities / selected_probabilities.sum(dim=-1, keepdim=True))
        selected_counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts)
        frequency = selected_counts.float() / (count * self.top_k)
        aux_loss = self.num_experts * (frequency.detach() * probabilities.mean(dim=0)).sum()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()

        if self.ep_rank is not None:
            combined = self._dispatch_experts(flat, indices, gates)
            kept_counts = selected_counts
            dropped_tokens = selected_counts.new_zeros(())
        else:
            capacity = (None if self.capacity_factor is None else
                        max(1, math.ceil(self.capacity_factor * count * self.top_k / self.num_experts)))
            combined = torch.zeros_like(flat, dtype=torch.float32)
            kept_gate_sum = torch.zeros(count, device=flat.device, dtype=torch.float32)
            kept_counts = []
            for expert_id, expert in enumerate(self.experts):
                token_indices, slots = (indices == expert_id).nonzero(as_tuple=True)
                if capacity is not None and token_indices.numel() > capacity:
                    # Keep the highest original router probabilities for this expert.
                    priority = probabilities[token_indices, expert_id].argsort(
                        descending=True, stable=True)[:capacity]
                    token_indices, slots = token_indices[priority], slots[priority]
                kept_counts.append(token_indices.numel())
                # Execute even for an empty selection: every rank keeps every expert in
                # the backward graph, as required by this DDP path's fixed parameter set.
                expert_values = expert(flat.index_select(0, token_indices)).float()
                weights = gates[token_indices, slots]
                combined = combined.index_add(0, token_indices, expert_values * weights[:, None])
                kept_gate_sum = kept_gate_sum.index_add(0, token_indices, weights)
            dropped_tokens = (kept_gate_sum == 0).sum()
            if self.top_k > 1:
                combined = combined / kept_gate_sum.clamp_min(1e-12)[:, None]
        return MoEOutput(
            hidden_states=combined.to(hidden_states.dtype).reshape(batch, length, width),
            aux_loss=aux_loss,
            selected_counts=selected_counts,
            kept_counts=(kept_counts if self.ep_rank is not None else
                         torch.tensor(kept_counts, dtype=torch.long, device=flat.device)),
            dropped_tokens=dropped_tokens,
            router_entropy=entropy,
            qb_beta_candidate=qb_beta_candidate,
        )
