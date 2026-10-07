#!/usr/bin/env python3
"""Two-rank expert-parallel MoE training: two of four experts per GPU.

The router and all non-expert weights are replicated. Selected tokens travel to
their expert owner with differentiable all-to-all and return to their source
rank for gated combination. Expert parameters are never gradient-reduced;
replicated parameters are reduced once after each global batch.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train import pretrain as single  # noqa: E402
from scripts.train import pretrain_ddp as ddp  # noqa: E402
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402


def shared_and_expert_parameters(model):
    shared, experts = [], []
    for name, parameter in model.named_parameters():
        (experts if ".mlp.experts." in name else shared).append(parameter)
    return shared, experts


@torch.no_grad()
def synchronize_shared_weights(shared):
    for parameter in shared:
        dist.broadcast(parameter, src=0)


@torch.no_grad()
def synchronize_shared_gradients(shared, *, bucket_bytes=32 * 1024**2):
    """SUM local global-loss contributions, without reducing expert gradients."""
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for parameter in bucket:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + count].reshape_as(parameter))
            offset += count
        bucket.clear()

    for parameter in shared:
        if parameter.grad is None:
            raise RuntimeError("EP 共享参数缺少梯度")
        bytes_here = parameter.grad.numel() * parameter.grad.element_size()
        if bucket and size + bytes_here > bucket_bytes:
            flush()
            size = 0
        bucket.append(parameter)
        size += bytes_here
    flush()


@torch.no_grad()
def clip_ep_grad_norm(shared, experts, max_norm, device):
    shared_sq = sum((parameter.grad.detach().double().square().sum() for parameter in shared),
                    torch.zeros((), dtype=torch.float64, device=device))
    local_expert_sq = sum((parameter.grad.detach().double().square().sum() for parameter in experts),
                          torch.zeros((), dtype=torch.float64, device=device))
    dist.all_reduce(local_expert_sq, op=dist.ReduceOp.SUM)
    norm = (shared_sq + local_expert_sq).sqrt()
    if not torch.isfinite(norm).item():
        raise RuntimeError("EP 梯度范数出现 NaN/Inf")
    scale = min(1.0, max_norm / (norm.item() + 1e-6))
    for parameter in (*shared, *experts):
        parameter.grad.mul_(scale)
    return norm.item()


def train_update(model, optimizer, dataset, global_indices, args, ctx, shared, experts,
                 routing_metrics=None):
    count = len(global_indices)
    local_indices = np.array_split(global_indices, ctx.world_size)[ctx.rank]
    rounds = math.ceil(math.ceil(count / ctx.world_size) / args.batch_size)
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = 0.0
    selected = torch.zeros((model.config.num_hidden_layers, model.config.num_experts),
                           dtype=torch.float64, device=ctx.device)
    entropy_sum = torch.zeros((), dtype=torch.float64, device=ctx.device)
    aux_sum = torch.zeros((), dtype=torch.float64, device=ctx.device)
    real_sequences = 0
    for index in range(rounds):
        part = local_indices[index * args.batch_size:(index + 1) * args.batch_size]
        batch = dataset.batch(part if len(part) else global_indices[:1], ctx.device)
        with single.precision_context(ctx.device, args.precision):
            output = model(input_ids=batch, labels=batch)
        objective = output.loss + args.moe_aux_loss_coef * output.aux_loss
        # EP sums shared gradients explicitly; each real sequence gets weight 1/N.
        (objective * (len(part) / count)).backward()
        local_loss_sum += output.loss.detach().item() * len(part)
        if routing_metrics is not None and len(part):
            selected += output.router_stats["selected_counts"].detach().double()
            entropy_sum += output.router_stats["router_entropy"].detach().double().mean() * len(part)
            aux_sum += output.aux_loss.detach().double() * len(part)
            real_sequences += len(part)

    total = torch.tensor(local_loss_sum, dtype=torch.float64, device=ctx.device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if not torch.isfinite(total).item():
        raise RuntimeError("EP 训练 loss 出现 NaN/Inf")
    synchronize_shared_gradients(shared)
    norm = clip_ep_grad_norm(shared, experts, args.max_grad_norm, ctx.device)
    optimizer.step()
    if routing_metrics is not None:
        dist.all_reduce(selected, op=dist.ReduceOp.SUM)
        summary = torch.stack((entropy_sum, aux_sum,
                               torch.tensor(real_sequences, dtype=torch.float64, device=ctx.device)))
        dist.all_reduce(summary, op=dist.ReduceOp.SUM)
        routing_metrics["router_entropy"] = (summary[0] / summary[2]).item()
        routing_metrics["router_aux_loss"] = (summary[1] / summary[2]).item()
        routing_metrics["router_dropped_assignment_fraction"] = 0.0
        routing_metrics["router_dropped_token_fraction"] = 0.0
        for layer in range(model.config.num_hidden_layers):
            for expert in range(model.config.num_experts):
                value = (selected[layer, expert] / selected[layer].sum()).item()
                routing_metrics[f"router/layer{layer}/expert{expert}_selected_fraction"] = value
                routing_metrics[f"router/layer{layer}/expert{expert}_kept_fraction"] = value
    return total.item() / count, norm


@torch.inference_mode()
def evaluate(model, dataset, args, ctx):
    """Every rank performs the same number of forwards for all-to-all ordering."""
    was_training = model.training
    model.eval()
    count = len(dataset) if args.eval_batches == 0 else min(len(dataset), args.eval_batches * args.batch_size)
    start, end = count * ctx.rank // 2, count * (ctx.rank + 1) // 2
    rounds = math.ceil(math.ceil(count / 2) / args.batch_size)
    loss_sum = 0.0
    try:
        for index in range(rounds):
            cursor = start + index * args.batch_size
            stop = min(cursor + args.batch_size, end)
            batch = dataset.batch(slice(cursor, stop) if cursor < stop else slice(0, 1), ctx.device)
            with single.precision_context(ctx.device, args.precision):
                loss = model(input_ids=batch, labels=batch).loss
            if cursor < stop:
                loss_sum += loss.item() * (stop - cursor)
        sums = torch.tensor([loss_sum, end - start], dtype=torch.float64, device=ctx.device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        if not torch.isfinite(sums).all().item() or sums[1].item() != count:
            raise RuntimeError("EP 验证 loss 或样本计数错误")
        return sums[0].item() / count, count
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx,
                    current_slot):
    """Write the inactive shard slot, then atomically point metadata at it."""
    slot = 0 if current_slot is None else 1 - current_slot
    shard_name = f"checkpoint_slot{slot}_rank{ctx.rank}.pt"
    shard_path = path.parent / shard_name
    error = None
    try:
        temporary = shard_path.with_suffix(".pt.tmp")
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": single.rng_state(ctx.device)}, temporary)
        temporary.replace(shard_path)
    except Exception as exc:
        error = f"rank {ctx.rank}: {type(exc).__name__}: {exc}"
    errors = [None] * ctx.world_size
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"EP 分片保存失败：{errors}")

    def write_metadata():
        metadata = {"format_version": 4, "model_config": model.config.to_dict(),
                    "contract": contract, "progress": dict(progress),
                    "scheduler_state_dict": scheduler.state_dict(), "slot": slot,
                    "shards": [f"checkpoint_slot{slot}_rank{rank}.pt" for rank in range(2)],
                    "tokenizer_path": str(tokenizer_path.resolve()),
                    "torch_version": str(torch.__version__)}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(metadata, temporary)
        temporary.replace(path)
    ddp.rank_zero_call(ctx, write_metadata)
    return slot


def load_checkpoint(path, contract, ctx):
    metadata = torch.load(path, map_location="cpu", weights_only=True)
    if metadata.get("format_version") != 4 or metadata.get("contract") != contract:
        raise ValueError("EP 恢复需要相同数据、模型、双卡拓扑和训练约定的 v4 checkpoint")
    shard = torch.load(path.parent / metadata["shards"][ctx.rank],
                       map_location="cpu", weights_only=True)
    return metadata, shard


def run(args, ctx):
    if ctx.world_size != 2:
        raise ValueError("此 EP 入口需要 torchrun 启动两个 rank")
    if args.precision == "bf16" and (ctx.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 需要支持该精度的 CUDA GPU；CPU 测试使用 fp32")
    if args.optimizer != "adamw" or args.record_update_norms:
        raise ValueError("首版 EP 仅支持 AdamW，不记录优化器更新范数")
    config = MiniLlamaConfig.from_dict(yaml.safe_load(args.model_config.read_text())["model_config"])
    if args.moe_top_k is not None or args.moe_capacity_factor is not None or args.moe_routing is not None:
        config = replace(config,
                             experts_per_token=args.moe_top_k or config.experts_per_token,
                             capacity_factor=(args.moe_capacity_factor if args.moe_capacity_factor is not None
                                              else config.capacity_factor),
                             moe_routing=args.moe_routing or config.moe_routing)
    if config.num_experts != 4 or config.moe_routing != "topk" or config.capacity_factor is not None:
        raise ValueError("首版 EP 需要四专家 top-k MoE，且不设容量上限")
    args.moe_aux_loss_coef = 0.01 if args.moe_aux_loss_coef is None else args.moe_aux_loss_coef
    datasets, fingerprint, tokenizer_path = single.load_data(args.data_dir, config)
    rows = len(datasets["train"]) if args.train_sequences is None else args.train_sequences
    if rows > len(datasets["train"]):
        raise ValueError("train-sequences 超过训练集序列数")
    sequence_length = fingerprint["sequence_length"]
    global_batch = args.batch_size * args.grad_accum_steps * 2
    total_steps = math.ceil(rows / global_batch) * args.epochs
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup-steps 必须小于完整训练步数")
    recipe_names = ("batch_size", "grad_accum_steps", "epochs", "learning_rate", "min_lr_ratio",
                    "warmup_steps", "weight_decay", "beta1", "beta2", "max_grad_norm", "seed",
                    "precision", "eval_batches")
    contract = {"model_config": config.to_dict(), "data": fingerprint,
                "recipe": {name: getattr(args, name) for name in recipe_names},
                "world_size": 2, "global_batch": global_batch, "device_type": ctx.device.type,
                "total_steps": total_steps, "parallelism": "ep2"}
    contract["recipe"]["moe_aux_loss_coef"] = args.moe_aux_loss_coef
    if args.train_sequences is not None:
        contract["recipe"]["train_sequences"] = args.train_sequences
    contracts = [None, None]
    dist.all_gather_object(contracts, contract)
    if contracts[0] != contracts[1]:
        raise ValueError("EP 两个 rank 的训练约定不一致")

    metadata, shard = load_checkpoint(args.resume, contract, ctx) if args.resume else (None, None)
    torch.manual_seed(args.seed)
    model = MiniLlamaForCausalLM(config).to(ctx.device)
    full_parameters = model.num_parameters()
    for layer in model.layers:
        layer.mlp.shard_experts(ctx.rank, 2)
    shared, experts = shared_and_expert_parameters(model)
    synchronize_shared_weights(shared)
    optimizer = single.make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda i: single.lr_factor(i, total_steps, args.warmup_steps, args.min_lr_ratio))
    progress = {"step": 0, "epoch": 0, "next_sequence": 0, "tokens_seen": 0,
                "last_train_loss": None, "initial_validation_loss": None,
                "validation_loss": None, "validation_step": None}
    if metadata:
        model.load_state_dict(shard["model_state_dict"], strict=True)
        optimizer.load_state_dict(shard["optimizer_state_dict"])
        scheduler.load_state_dict(metadata["scheduler_state_dict"])
        progress = metadata["progress"]
        single.validate_progress(progress, rows, global_batch, args.epochs)
        if scheduler.last_epoch != progress["step"]:
            raise ValueError("EP 学习率进度与 checkpoint step 不一致")
        if progress["tokens_seen"] != (progress["epoch"] * rows + progress["next_sequence"]) * sequence_length:
            raise ValueError("EP tokens_seen 与全局数据游标不一致")
    stop_step = min(total_steps, args.stop_after_steps or total_steps)
    if stop_step <= progress["step"]:
        raise ValueError("停止步数必须大于已完成步数")
    output = (args.output_dir or args.resume.parent).resolve()
    if args.resume and output != args.resume.resolve().parent:
        raise ValueError("EP 恢复必须写回原输出目录")
    if not args.resume:
        ddp.rank_zero_call(ctx, lambda: output.mkdir(parents=True, exist_ok=False))
    checkpoint_path = output / "checkpoint.pt"
    writer = None
    try:
        if ctx.rank == 0:
            writer = SummaryWriter(str(output / "tensorboard"),
                                   purge_step=progress["step"] + 1 if metadata else None)
        if shard:
            single.restore_rng(shard["rng_state"], ctx.device)
            del shard
        started = time.perf_counter()
        start_step = progress["step"]
        current_slot = metadata["slot"] if metadata else None
        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)
        if ctx.rank == 0:
            print(f"EP world_size=2 experts_per_rank=2 total_parameters={full_parameters:,} "
                  f"local_parameters={model.num_parameters():,} global_batch={global_batch} "
                  f"total_steps={total_steps} start_step={start_step} stop_step={stop_step}", flush=True)

        def validate():
            loss, count = evaluate(model, datasets["validation"], args, ctx)
            progress["validation_loss"] = loss
            progress["validation_step"] = progress["step"]
            if progress["initial_validation_loss"] is None:
                progress["initial_validation_loss"] = loss
            if writer:
                writer.add_scalar("validation/loss", loss, progress["step"])
                writer.add_scalar("validation/prediction_tokens", count * (sequence_length - 1),
                                  progress["step"])
                print(f"validation step={progress['step']} loss={loss:.6f} sequences={count}", flush=True)

        if progress["step"] == 0:
            validate()
            current_slot = save_checkpoint(checkpoint_path, model, optimizer, scheduler, progress,
                                           contract, tokenizer_path, ctx, current_slot)
        model.train()
        order_epoch = None
        while progress["step"] < stop_step:
            if order_epoch != progress["epoch"]:
                order_epoch = progress["epoch"]
                order = single.epoch_order(rows, args.seed, order_epoch)
            cursor = progress["next_sequence"]
            end = min(cursor + global_batch, rows)
            indices = order[cursor:end]
            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)
            dist.barrier()
            update_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            next_step = progress["step"] + 1
            routing = ({} if next_step == 1 or next_step % args.log_every == 0 or next_step == stop_step
                       else None)
            loss, norm = train_update(model, optimizer, datasets["train"], indices, args, ctx,
                                      shared, experts, routing)
            scheduler.step()
            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)
            elapsed = torch.tensor(time.perf_counter() - update_started, dtype=torch.float64,
                                   device=ctx.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            progress["step"] = next_step
            progress["tokens_seen"] += len(indices) * sequence_length
            progress["next_sequence"] = end
            progress["last_train_loss"] = loss
            if end == rows:
                progress["epoch"] += 1
                progress["next_sequence"] = 0
            metrics = {"loss": loss, "learning_rate": learning_rate, "gradient_norm": norm,
                       "tokens_seen": progress["tokens_seen"],
                       "tokens_per_second": len(indices) * sequence_length / elapsed.item()}
            if routing is not None:
                metrics.update(routing)
            if ctx.device.type == "cuda":
                memory = torch.tensor(torch.cuda.max_memory_allocated(ctx.device) / 1024**3,
                                      device=ctx.device)
                memories = [torch.zeros_like(memory), torch.zeros_like(memory)]
                dist.all_gather(memories, memory)
                metrics.update({f"peak_memory_gib_rank{rank}": value.item()
                                for rank, value in enumerate(memories)})
                metrics["peak_memory_gib"] = max(value.item() for value in memories)
            if writer:
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, next_step)
                if next_step == 1 or next_step % args.log_every == 0 or next_step == stop_step:
                    print(f"step={next_step}/{total_steps} loss={loss:.6f} lr={learning_rate:.3e} "
                          f"grad_norm={norm:.4f} global_tokens/s={metrics['tokens_per_second']:.0f}",
                          flush=True)
            if next_step % args.eval_every == 0 or next_step == stop_step:
                validate()
            if next_step % args.save_every == 0 or next_step == stop_step:
                current_slot = save_checkpoint(checkpoint_path, model, optimizer, scheduler, progress,
                                               contract, tokenizer_path, ctx, current_slot)
                if writer:
                    writer.flush()

        session_seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64,
                                       device=ctx.device)
        dist.all_reduce(session_seconds, op=dist.ReduceOp.MAX)
        summary = {"status": "complete" if progress["step"] == total_steps else "paused",
                   "model_parameters": full_parameters, "local_model_parameters": model.num_parameters(),
                   "progress": progress, "contract": contract, "device": str(ctx.device),
                   "world_size": 2, "start_step": start_step,
                   "session_elapsed_seconds": round(session_seconds.item(), 2),
                   "prediction_tokens_seen": progress["tokens_seen"] // sequence_length * (sequence_length - 1),
                   "checkpoint": str(checkpoint_path), "tensorboard": str(output / "tensorboard")}

        def write_summary():
            temporary = output / "summary.json.tmp"
            temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output / "summary.json")
            print(f"{summary['status']}: {checkpoint_path}", flush=True)
        ddp.rank_zero_call(ctx, write_summary)
    finally:
        if writer:
            writer.close()


def main():
    args = ddp.parse_args(ddp.build_parser(description=__doc__))
    torch.set_num_threads(args.cpu_threads)
    ctx = ddp.setup_distributed(args.device)
    try:
        run(args, ctx)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
