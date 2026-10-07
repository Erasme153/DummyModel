#!/usr/bin/env python3
"""M3 FSDP2 训练：与 pretrain_ddp.py 保持相同的数据、batch 和 LR 约定。

双卡示例：torchrun --standalone --nproc-per-node=2 scripts/train/pretrain_fsdp.py \
  --device cuda --batch-size 4 --grad-accum-steps 2 \
  --output-dir runs/m03_99m_fsdp2 --stop-after-steps 100

checkpoint/ 是分片训练状态目录；--resume 指向该目录。可选的
--export-full-checkpoint 会另存仅用于推理的 checkpoint_full.pt。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_state_dict, set_model_state_dict, set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.utils.tensorboard import SummaryWriter
import yaml

try:  # PyTorch 2.5.1 exposes FSDP2 through the composable namespace.
    from torch.distributed.fsdp import fully_shard
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train import pretrain as single  # noqa: E402
from scripts.train import pretrain_ddp as ddp  # noqa: E402
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402


def parse_args():
    parser = ddp.build_parser(description=__doc__, device_help="仅支持 cuda；由 LOCAL_RANK 绑定 GPU")
    parser.add_argument("--export-full-checkpoint", action="store_true",
                        help="结束时另外导出可供现有推理脚本加载的完整模型权重")
    args = ddp.parse_args(parser)
    if args.train_sequences is not None:
        parser.error("--train-sequences 目前仅 DDP 入口支持")
    if args.optimizer != "adamw" or args.record_update_norms:
        parser.error("FSDP2 入口暂不支持 Muon/Hyperball 或更新范数记录；请使用 DDP 入口")
    return args


def shard_model(model, world_size):
    mesh = init_device_mesh("cuda", (world_size,))
    for block in model.layers:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def clip_sharded_grad_norm_(model, max_norm):
    """先汇总各 rank 的梯度分片平方和，再统一裁剪。"""
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.grad is not None]
    local_squares = torch.zeros(
        (), dtype=torch.float64, device=torch.device("cuda", torch.cuda.current_device())
    )
    for name, parameter in parameters:
        gradient = parameter.grad
        if not isinstance(gradient, DTensor):
            raise TypeError(f"FSDP2 梯度预期为 DTensor：{name} ({type(gradient).__name__})")
        local = gradient.to_local()
        local_squares += local.double().square().sum()
    dist.all_reduce(local_squares, op=dist.ReduceOp.SUM)
    norm = local_squares.sqrt().item()
    if not math.isfinite(norm):
        raise RuntimeError("梯度范数出现 NaN/Inf，停止更新")
    factor = min(1.0, max_norm / (norm + 1e-6))
    if factor < 1.0:
        for _, parameter in parameters:
            parameter.grad.mul_(factor)
    return norm


def train_update(model, optimizer, dataset, global_indices, args, ctx):
    count = len(global_indices)
    if count == 0:
        raise ValueError("全局 batch 不能为空")
    local_indices = np.array_split(global_indices, ctx.world_size)[ctx.rank]
    rounds = math.ceil(math.ceil(count / ctx.world_size) / args.batch_size)
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = 0.0
    try:
        for index in range(rounds):
            part = local_indices[index * args.batch_size:(index + 1) * args.batch_size]
            batch = dataset.batch(part if len(part) else global_indices[:1], ctx.device)
            # FSDP2 用 reduce-scatter 平均各 rank 梯度。非最后一轮只累积，
            # 最后一轮同步；零权重 dummy 保证尾部所有 rank 的调用次数一致。
            model.set_requires_gradient_sync(index == rounds - 1)
            with single.precision_context(ctx.device, args.precision):
                loss = model(input_ids=batch, labels=batch).loss
            if loss is None:
                raise RuntimeError("模型没有返回 causal LM loss")
            (loss * (ctx.world_size * len(part) / count)).backward()
            local_loss_sum += loss.detach().item() * len(part)
    finally:
        model.set_requires_gradient_sync(True)

    total = torch.tensor(local_loss_sum, dtype=torch.float64, device=ctx.device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if not torch.isfinite(total).item():
        raise RuntimeError("训练 loss 出现 NaN/Inf，停止更新")
    norm = clip_sharded_grad_norm_(model, args.max_grad_norm)
    optimizer.step()
    return total.item() / count, norm


@torch.no_grad()
def evaluate(model, dataset, args, ctx):
    """无重复验证；no_grad 避免 FSDP2 缓存 inference tensors 影响下一次 backward。"""
    was_training = model.training
    model.eval()
    count = len(dataset) if args.eval_batches == 0 else min(len(dataset), args.eval_batches * args.batch_size)
    start = count * ctx.rank // ctx.world_size
    end = count * (ctx.rank + 1) // ctx.world_size
    rounds = math.ceil(math.ceil(count / ctx.world_size) / args.batch_size)
    loss_sum = 0.0
    try:
        for index in range(rounds):
            left = start + index * args.batch_size
            right = min(left + args.batch_size, end)
            batch = dataset.batch(slice(left, right) if left < end else slice(0, 1), ctx.device)
            with single.precision_context(ctx.device, args.precision):
                loss = model(input_ids=batch, labels=batch).loss
            if left < end:
                loss_sum += loss.item() * len(batch)
        sums = torch.tensor([loss_sum, end - start], dtype=torch.float64, device=ctx.device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        if count == 0 or not torch.isfinite(sums).all().item() or sums[1].item() != count:
            raise RuntimeError("验证 loss 非有限或样本计数错误")
        return sums[0].item() / count, count
    finally:
        model.train(was_training)


def _initialize_adamw_state(optimizer):
    """DCP 加载需预分配目的张量；AdamW 的 moment 原本在首次 step 才创建。"""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            state = optimizer.state[parameter]
            if state:
                continue
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)


def checkpoint_state_for_save(model, optimizer, step):
    # PyTorch 2.5 的 get_state_dict 会为尚未更新的 AdamW 懒初始化状态并把
    # step 推到 1。第 0 步不保存空优化器状态，避免第一次真实更新从 step 2 开始。
    if step == 0:
        return {"model": get_model_state_dict(model)}
    model_state, optimizer_state = get_state_dict(model, optimizer)
    return {"model": model_state, "optimizer": optimizer_state}


def load_metadata(path, contract, ctx):
    path = path.resolve()
    if not path.is_dir() and path.name == "checkpoint":
        fallback = path.with_name("checkpoint.prev")
        if fallback.is_dir():
            path = fallback
    if not path.is_dir() or not (path / "trainer.pt").is_file():
        raise ValueError(f"需要 FSDP2 checkpoint 目录：{path}")
    checkpoint = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 3 or checkpoint.get("contract") != contract:
        raise ValueError("恢复失败：需要本 FSDP2 脚本的 v3 checkpoint，且分片方式、卡数、模型、数据和训练约定一致")
    if len(checkpoint.get("rng_states", [])) != ctx.world_size:
        raise ValueError("checkpoint 缺少各 rank 的 RNG 状态")
    return checkpoint, path


def load_training_state(path, model, optimizer, scheduler, checkpoint):
    if checkpoint["progress"]["step"] == 0:
        state = {"model": get_model_state_dict(model)}
        dcp.load(state, checkpoint_id=str(path))
        set_model_state_dict(model, state["model"])
    else:
        _initialize_adamw_state(optimizer)
        model_state, optimizer_state = get_state_dict(model, optimizer)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=str(path))
        set_state_dict(model, optimizer, model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def save_checkpoint(path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx):
    """所有 rank 保存分片；仅在完整写入后切换可恢复 checkpoint。"""
    temporary = path.with_name("checkpoint.tmp")
    previous = path.with_name("checkpoint.prev")

    def clear_stale():
        if temporary.exists():
            shutil.rmtree(temporary)
    ddp.rank_zero_call(ctx, clear_stale)

    dcp.save(checkpoint_state_for_save(model, optimizer, progress["step"]),
             checkpoint_id=str(temporary))
    local_rng = single.rng_state(ctx.device)
    rng_states = [None] * ctx.world_size
    dist.all_gather_object(rng_states, local_rng)

    def commit():
        metadata = {"format_version": 3, "parallelism": "fsdp2", "contract": contract,
                    "progress": dict(progress), "scheduler_state_dict": scheduler.state_dict(),
                    "rng_states": rng_states, "model_config": contract["model_config"],
                    "tokenizer_path": str(tokenizer_path.resolve()), "torch_version": str(torch.__version__)}
        torch.save(metadata, temporary / "trainer.pt")
        if path.exists():
            if previous.exists():
                shutil.rmtree(previous)
            path.rename(previous)
        temporary.rename(path)
        if previous.exists():
            shutil.rmtree(previous)
    ddp.rank_zero_call(ctx, commit)


def export_full_checkpoint(path, model, config, progress, tokenizer_path, ctx):
    # 仅在明确请求时聚合完整权重；不把优化器复制到单文件推理产物。
    weights = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

    def write():
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"format_version": 3, "model_config": config.to_dict(),
                    "model_state_dict": weights, "progress": dict(progress),
                    "tokenizer_path": str(tokenizer_path.resolve())}, temporary)
        temporary.replace(path)
    ddp.rank_zero_call(ctx, write)


def run(args, ctx):
    if ctx.world_size < 2 or ctx.device.type != "cuda":
        raise ValueError("FSDP2 入口需要 torchrun 启动至少两个 CUDA 进程")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 需要支持它的 CUDA GPU")
    config = MiniLlamaConfig.from_dict(yaml.safe_load(args.model_config.read_text())["model_config"])
    if config.num_experts:
        raise ValueError("MoE 训练请使用 pretrain_ddp.py；FSDP2 入口尚未汇总路由辅助损失")
    datasets, fingerprint, tokenizer_path = single.load_data(args.data_dir, config)
    sequence_length = fingerprint["sequence_length"]
    rows = len(datasets["train"])
    global_batch = args.batch_size * args.grad_accum_steps * ctx.world_size
    total_steps = math.ceil(rows / global_batch) * args.epochs
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup-steps 必须小于完整训练步数")
    recipe_names = ("batch_size", "grad_accum_steps", "epochs", "learning_rate", "min_lr_ratio",
                    "warmup_steps", "weight_decay", "beta1", "beta2", "max_grad_norm",
                    "seed", "precision", "eval_batches")
    contract = {"model_config": config.to_dict(), "data": fingerprint,
                "recipe": {name: getattr(args, name) for name in recipe_names},
                "world_size": ctx.world_size, "global_batch": global_batch,
                "device_type": ctx.device.type, "total_steps": total_steps,
                "parallelism": "fsdp2", "shard_policy": "blocks"}
    contracts = [None] * ctx.world_size
    dist.all_gather_object(contracts, contract)
    if any(item != contract for item in contracts):
        raise ValueError("各 rank 的模型、数据或训练约定不一致")
    checkpoint, resume_path = load_metadata(args.resume, contract, ctx) if args.resume else (None, None)

    torch.manual_seed(args.seed)
    model = MiniLlamaForCausalLM(config).to(ctx.device)
    parameters = model.num_parameters()
    shard_model(model, ctx.world_size)
    optimizer = single.make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda i: single.lr_factor(i, total_steps, args.warmup_steps, args.min_lr_ratio))
    progress = {"step": 0, "epoch": 0, "next_sequence": 0, "tokens_seen": 0,
                "last_train_loss": None, "initial_validation_loss": None,
                "validation_loss": None, "validation_step": None}
    if checkpoint:
        load_training_state(resume_path, model, optimizer, scheduler, checkpoint)
        progress = checkpoint["progress"]
        single.validate_progress(progress, rows, global_batch, args.epochs)
        if scheduler.last_epoch != progress["step"]:
            raise ValueError("学习率进度与 step 不一致")
        if progress["tokens_seen"] != (progress["epoch"] * rows + progress["next_sequence"]) * sequence_length:
            raise ValueError("tokens_seen 与全局数据游标不一致")
    stop_step = min(total_steps, args.stop_after_steps or total_steps)
    if stop_step <= progress["step"]:
        raise ValueError("停止步数必须大于 checkpoint 已完成的步数")
    output = (args.output_dir or args.resume.parent).resolve()
    if args.resume is None or output != args.resume.resolve().parent:
        ddp.rank_zero_call(ctx, lambda: output.mkdir(parents=True, exist_ok=False))
    checkpoint_path = output / "checkpoint"
    writer = None
    try:
        if ctx.rank == 0:
            writer = SummaryWriter(str(output / "tensorboard"),
                                   purge_step=progress["step"] + 1 if checkpoint else None)
        if checkpoint:
            single.restore_rng(checkpoint["rng_states"][ctx.rank], ctx.device)
            del checkpoint
        started = time.perf_counter()
        start_step = progress["step"]
        torch.cuda.reset_peak_memory_stats(ctx.device)
        if ctx.rank == 0:
            print(f"parameters={parameters:,} world_size={ctx.world_size} global_batch={global_batch} "
                  f"total_steps={total_steps} start_step={start_step} stop_step={stop_step} "
                  f"precision={args.precision} shard_policy=blocks", flush=True)

        def validate():
            loss, count = evaluate(model, datasets["validation"], args, ctx)
            progress["validation_loss"] = loss
            progress["validation_step"] = progress["step"]
            if progress["initial_validation_loss"] is None:
                progress["initial_validation_loss"] = loss
            if writer:
                writer.add_scalar("validation/loss", loss, progress["step"])
                writer.add_scalar("validation/prediction_tokens", count * (sequence_length - 1), progress["step"])
                print(f"validation step={progress['step']} loss={loss:.6f} sequences={count}", flush=True)

        if progress["step"] == 0:
            validate()
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx)
        model.train()
        order_epoch = None
        while progress["step"] < stop_step:
            if order_epoch != progress["epoch"]:
                order_epoch = progress["epoch"]
                order = single.epoch_order(rows, args.seed, order_epoch)
            cursor = progress["next_sequence"]
            end = min(cursor + global_batch, rows)
            indices = order[cursor:end]
            torch.cuda.synchronize(ctx.device)
            update_started = time.perf_counter()
            dist.barrier()
            learning_rate = optimizer.param_groups[0]["lr"]
            loss, norm = train_update(model, optimizer, datasets["train"], indices, args, ctx)
            scheduler.step()
            torch.cuda.synchronize(ctx.device)
            elapsed = torch.tensor(time.perf_counter() - update_started, dtype=torch.float64, device=ctx.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            progress["step"] += 1
            progress["tokens_seen"] += len(indices) * sequence_length
            progress["next_sequence"] = end
            progress["last_train_loss"] = loss
            if end == rows:
                progress["epoch"] += 1
                progress["next_sequence"] = 0
            step = progress["step"]
            metrics = {"loss": loss, "learning_rate": learning_rate, "gradient_norm": norm,
                       "tokens_seen": progress["tokens_seen"],
                       "tokens_per_second": len(indices) * sequence_length / elapsed.item()}
            memory = torch.tensor(torch.cuda.max_memory_allocated(ctx.device) / 1024**3, device=ctx.device)
            memories = [torch.zeros_like(memory) for _ in range(ctx.world_size)]
            dist.all_gather(memories, memory)
            metrics.update({f"peak_memory_gib_rank{r}": value.item() for r, value in enumerate(memories)})
            metrics["peak_memory_gib"] = max(value.item() for value in memories)
            if writer:
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, step)
                if step == 1 or step % args.log_every == 0 or step == stop_step:
                    print(f"step={step}/{total_steps} loss={loss:.6f} lr={learning_rate:.3e} "
                          f"grad_norm={norm:.4f} global_tokens/s={metrics['tokens_per_second']:.0f}", flush=True)
            if step % args.eval_every == 0 or step == stop_step:
                validate()
            if step % args.save_every == 0 or step == stop_step:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx)
                if writer:
                    writer.flush()

        if args.export_full_checkpoint:
            export_full_checkpoint(output / "checkpoint_full.pt", model, config, progress, tokenizer_path, ctx)
        session_seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=ctx.device)
        dist.all_reduce(session_seconds, op=dist.ReduceOp.MAX)
        summary = {"status": "complete" if progress["step"] == total_steps else "paused",
                   "model_parameters": parameters, "progress": progress, "contract": contract,
                   "device": str(ctx.device), "world_size": ctx.world_size, "start_step": start_step,
                   "session_elapsed_seconds": round(session_seconds.item(), 2),
                   "prediction_tokens_seen": progress["tokens_seen"] // sequence_length * (sequence_length - 1),
                   "checkpoint": str(checkpoint_path), "tensorboard": str(output / "tensorboard")}
        if args.export_full_checkpoint:
            summary["full_checkpoint"] = str(output / "checkpoint_full.pt")

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
    args = parse_args()
    if torch.device(args.device).type != "cuda":
        raise ValueError("FSDP2 入口仅支持 CUDA；请使用 --device cuda")
    torch.set_num_threads(args.cpu_threads)
    ctx = ddp.setup_distributed(args.device)
    try:
        run(args, ctx)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
