#!/usr/bin/env python3
"""M9: two-rank group-relative policy optimization on GSM8K."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402
from scripts.eval import grpo_baseline as baseline  # noqa: E402
from scripts.train import pretrain as single  # noqa: E402
from scripts.train import pretrain_ddp as ddp  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/m09_gsm8k/main")
    result.add_argument("--ultrafeedback-dir", type=Path,
                        default=ROOT / "data/raw/m09_ultrafeedback/data")
    result.add_argument("--init-checkpoint", type=Path,
                        default=ROOT / "runs/m09_213m_gsm8k_sft/checkpoint.pt")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--resume", type=Path)
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--max-length", type=int, default=2048)
    result.add_argument("--holdout-size", type=int, default=256)
    result.add_argument("--max-new-tokens", type=int, default=256)
    result.add_argument("--group-size", type=int, default=8)
    result.add_argument("--batch-size", type=int, default=1,
                        help="每 rank 每 micro-batch 的题目数")
    result.add_argument("--grad-accum-steps", type=int, default=4)
    result.add_argument("--policy-epochs", type=int, default=2,
                        help="同一批采样的 PPO/GRPO 更新次数")
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--temperature", type=float, default=0.7)
    result.add_argument("--top-k", type=int, default=50)
    result.add_argument("--top-p", type=float, default=0.95)
    result.add_argument("--learning-rate", type=float, default=5e-6)
    result.add_argument("--kl-coef", type=float, default=0.01)
    result.add_argument("--clip-eps", type=float, default=0.2)
    result.add_argument("--min-lr-ratio", type=float, default=0.1)
    result.add_argument("--warmup-ratio", type=float, default=0.03)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--beta1", type=float, default=0.9)
    result.add_argument("--beta2", type=float, default=0.95)
    result.add_argument("--max-grad-norm", type=float, default=1.0)
    result.add_argument("--save-every", type=int, default=50)
    result.add_argument("--log-every", type=int, default=10)
    result.add_argument("--stop-after-steps", type=int,
                        help="采样轮数；每轮执行 policy-epochs 次优化器更新")
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--cpu-threads", type=int, default=4)
    args = result.parse_args()
    for name in ("max_length", "holdout_size", "max_new_tokens", "batch_size",
                 "grad_accum_steps", "policy_epochs", "epochs", "save_every",
                 "log_every", "cpu_threads"):
        if getattr(args, name) <= 0:
            result.error(f"{name} 必须为正数")
    if args.group_size < 2 or args.seed < 0 or args.top_k < 0:
        result.error("group-size 至少为 2；seed/top-k 不能为负数")
    if args.max_new_tokens >= args.max_length:
        result.error("max-new-tokens 必须小于 max-length")
    for name in ("temperature", "learning_rate", "clip_eps", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            result.error(f"{name} 必须为有限正数")
    if not 0 < args.top_p <= 1 or not math.isfinite(args.top_p):
        result.error("top-p 必须在 (0,1] 内")
    if args.kl_coef < 0 or not math.isfinite(args.kl_coef) or \
            args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        result.error("kl-coef/weight-decay 必须为有限非负数")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1 or \
            not all(0 <= value < 1 for value in (args.beta1, args.beta2)):
        result.error("warmup-ratio/min-lr-ratio/beta1/beta2 超出范围")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        result.error("stop-after-steps 必须为正数")
    if args.resume and args.output_dir:
        result.error("--resume 使用原输出目录，不再指定 --output-dir")
    if not args.audit_only and not args.resume and not args.output_dir:
        result.error("新训练必须指定 --output-dir")
    return args


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """每行是同一题的采样；同分组精确返回零优势。"""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards 必须为 [题目数, group-size>=2]")
    values = rewards.float()
    centered = values - values.mean(dim=1, keepdim=True)
    scale = torch.sqrt((centered * centered).mean(dim=1, keepdim=True))
    return torch.where(scale > 0, centered / scale.clamp_min(1e-8),
                       torch.zeros_like(centered))


def collate_rollouts(rows, device, eos):
    """rows: (prompt_ids, completion_ids, reward)，只监督采样到的 completion。"""
    length = max(len(prompt) + len(completion) for prompt, completion, _ in rows)
    ids = torch.full((len(rows), length), eos, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, (prompt, completion, _) in enumerate(rows):
        end = len(prompt) + len(completion)
        ids[index, :end] = torch.tensor((*prompt, *completion), dtype=torch.long)
        labels[index, len(prompt):end] = torch.tensor(completion, dtype=torch.long)
        mask[index, :end] = True
    return ids.to(device), labels.to(device), mask.to(device)


def token_logps(logits, labels):
    """返回 [B,T-1]，prompt/padding 的位置为零。"""
    targets = labels[:, 1:]
    valid = targets != -100
    logps = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return logps.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1) * valid


def grpo_objective(current, old, reference, labels, advantages, *, clip_eps, kl_coef):
    if current.shape != old.shape or current.shape != reference.shape or \
            current.shape != labels[:, 1:].shape or advantages.shape != current.shape[:1]:
        raise ValueError("GRPO log-prob、标签和优势形状不一致")
    valid = labels[:, 1:] != -100
    if not valid.any(dim=1).all():
        raise ValueError("GRPO 每条序列必须有 completion token")
    ratio = torch.exp((current - old).clamp(-20, 20))
    advantage = advantages[:, None]
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
    delta = reference - current
    kl = torch.exp(delta.clamp(-20, 20)) - delta - 1
    per_sequence = ((-surrogate + kl_coef * kl) * valid).sum(1) / valid.sum(1)
    clipped = (((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)) & valid).sum()
    return per_sequence, (kl * valid).sum() / valid.sum(), clipped / valid.sum()


def make_rollout(policy, reference, examples, indices, tokenizer, config, args, ctx,
                 epoch):
    """每 rank 为自己的题目生成固定 on-policy 采样，并缓存旧/参考 log-prob。"""
    policy.eval()
    batches = []
    local_count = args.batch_size * args.grad_accum_steps
    local = indices[ctx.rank * local_count:(ctx.rank + 1) * local_count]
    totals = torch.zeros(4, dtype=torch.float64, device=ctx.device)
    for micro in range(args.grad_accum_steps):
        rows = []
        selected = local[micro * args.batch_size:(micro + 1) * args.batch_size]
        for index in selected:
            example = examples[int(index)]
            generator = torch.Generator(device=ctx.device).manual_seed(
                args.seed + epoch * 10000019 + example["source_index"])
            texts, _, completions = baseline.generate_group(
                policy, example["prompt_ids"], tokenizer, group_size=args.group_size,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p, eos=config.eos_token_id,
                generator=generator, device=ctx.device, precision=args.precision,
                return_token_ids=True)
            gold = Decimal(example["answer"])
            rewards = [float(baseline.final_number(text) == gold) for text in texts]
            rows.extend((example["prompt_ids"], completion, reward)
                        for completion, reward in zip(completions, rewards))
            positives = sum(rewards)
            totals += torch.tensor([1, positives, int(0 < positives < args.group_size),
                                    sum(baseline.final_number(text) is not None
                                        for text in texts)],
                                   dtype=torch.float64, device=ctx.device)
        ids, labels, mask = collate_rollouts(rows, ctx.device, config.eos_token_id)
        rewards = torch.tensor([row[2] for row in rows], device=ctx.device).reshape(
            args.batch_size, args.group_size)
        advantages = group_advantages(rewards).reshape(-1)
        with torch.no_grad():
            with single.precision_context(ctx.device, args.precision):
                old = token_logps(policy(ids, attention_mask=mask).logits, labels)
                ref = token_logps(reference(ids, attention_mask=mask).logits, labels)
        batches.append((ids, labels, mask, advantages, old.detach(), ref.detach()))
    if ctx.distributed:
        dist.all_reduce(totals)
    return batches, totals


def update_policy(model, optimizer, batches, args, ctx):
    model.eval()  # 采样和 log-prob 计算都关闭 dropout。
    optimizer.zero_grad(set_to_none=True)
    total_samples = args.batch_size * args.grad_accum_steps * ctx.world_size * args.group_size
    totals = torch.zeros(3, dtype=torch.float64, device=ctx.device)
    for micro, (ids, labels, mask, advantages, old, ref) in enumerate(batches):
        sync = nullcontext() if micro == len(batches) - 1 or not ctx.distributed else model.no_sync()
        with sync:
            with single.precision_context(ctx.device, args.precision):
                current = token_logps(model(ids, attention_mask=mask).logits, labels)
                losses, kl, clipped = grpo_objective(
                    current, old, ref, labels, advantages,
                    clip_eps=args.clip_eps, kl_coef=args.kl_coef)
            if not torch.isfinite(losses).all():
                raise RuntimeError("GRPO loss 非有限")
            (losses.sum() * ctx.world_size / total_samples).backward()
        totals += torch.tensor([losses.detach().sum().item(),
                                kl.detach().item() * len(losses),
                                clipped.detach().item() * len(losses)],
                               dtype=torch.float64, device=ctx.device)
    norm = clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if ctx.distributed:
        dist.all_reduce(totals)
    return {"loss": (totals[0] / total_samples).item(),
            "kl": (totals[1] / total_samples).item(),
            "clip_fraction": (totals[2] / total_samples).item(),
            "gradient_norm": float(norm)}


def save(path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx):
    rng = single.rng_state(ctx.device)
    states = [None] * ctx.world_size
    if ctx.distributed:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng

    def write():
        value = {"format_version": 6, "model_config": ddp.unwrap(model).config.to_dict(),
                 "model_state_dict": ddp.unwrap(model).state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "progress": dict(progress), "contract": contract, "rng_states": states,
                 "tokenizer_path": str(tokenizer_path.resolve()),
                 "torch_version": str(torch.__version__)}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(value, temporary)
        temporary.replace(path)
    ddp.rank_zero_call(ctx, write)


def run(args, ctx):
    if not args.audit_only and args.precision == "bf16" and (
            ctx.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 需要支持该精度的 CUDA GPU；CPU 检查请用 --precision fp32")
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True,
                        mmap=True)
    if source.get("format_version") != 3 or source.get("contract", {}).get("world_size") != ctx.world_size or \
            source.get("contract", {}).get("data", {}).get("dataset") != "gsm8k":
        raise ValueError("GRPO 起点需要同卡数的 GSM8K SFT v3 checkpoint")
    config = MiniLlamaConfig.from_dict(source["model_config"])
    if config.num_experts or args.max_length > config.max_position_embeddings:
        raise ValueError("GRPO 仅支持 dense 模型和不超过模型上下文的长度")
    tokenizer_path = Path(source["tokenizer_path"])
    tokenizer_sha = single.file_sha256(tokenizer_path)
    if tokenizer_sha != source["contract"]["data"]["tokenizer_sha256"]:
        raise ValueError("GRPO tokenizer 指纹不符")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("GRPO tokenizer 与模型词表不符")
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    datasets, audit = baseline.load_data(
        args.data_dir, args.ultrafeedback_dir, tokenizer, config,
        holdout_size=args.holdout_size, max_new_tokens=args.max_new_tokens,
        seed=2026, train_only=True)
    source_data = source["contract"]["data"]
    if (audit["holdout_source_indices_sha256"] != source_data["holdout_source_indices_sha256"]
            or audit["source_sha256"] != source_data["source_sha256"]
            or audit["test_questions_sha256"] != source_data["test_questions_sha256"]):
        raise ValueError("GRPO 训练/留出数据与 SFT 起点不一致")
    if any(len(row["prompt_ids"]) + args.max_new_tokens > args.max_length
           for row in datasets["train"]):
        raise ValueError("GRPO prompt+max-new-tokens 超过 max-length")
    global_batch = args.batch_size * args.grad_accum_steps * ctx.world_size
    steps_per_epoch = len(datasets["train"]) // global_batch
    if not steps_per_epoch:
        raise ValueError("GRPO 训练题目不足一个全局 batch")
    total_steps = steps_per_epoch * args.epochs
    total_updates = total_steps * args.policy_epochs
    warmup_steps = round(total_updates * args.warmup_ratio)
    source_sha = single.file_sha256(args.init_checkpoint)
    contract = {"model_config": config.to_dict(),
                "data": {"tokenizer_sha256": tokenizer_sha,
                         "source_sha256": audit["source_sha256"],
                         "holdout_source_indices_sha256": audit["holdout_source_indices_sha256"],
                         "test_questions_sha256": audit["test_questions_sha256"],
                         "split_seed": 2026},
                "recipe": {name: getattr(args, name) for name in
                           ("max_length", "max_new_tokens", "holdout_size", "group_size",
                            "batch_size", "grad_accum_steps", "policy_epochs", "epochs",
                            "temperature", "top_k", "top_p", "learning_rate", "kl_coef",
                            "clip_eps", "min_lr_ratio", "warmup_ratio", "weight_decay",
                            "beta1", "beta2", "max_grad_norm", "seed", "precision")},
                "world_size": ctx.world_size, "global_batch": global_batch,
                "total_steps": total_steps, "total_optimizer_updates": total_updates,
                "source_checkpoint": str(args.init_checkpoint.resolve()),
                "source_checkpoint_sha256": source_sha,
                "reference_checkpoint": str(args.init_checkpoint.resolve())}
    if args.audit_only:
        if ctx.rank == 0:
            print(json.dumps({"audit": audit, "global_batch": global_batch,
                              "steps_per_epoch": steps_per_epoch,
                              "total_optimizer_updates": total_updates,
                              "dropped_epoch_tail": len(datasets["train"]) % global_batch},
                             indent=2, ensure_ascii=False))
        return
    checkpoint = (torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
                  if args.resume else None)
    if checkpoint and (checkpoint.get("format_version") != 6 or
                       checkpoint.get("contract") != contract):
        raise ValueError("GRPO 恢复约定与 checkpoint 不一致")
    torch.manual_seed(args.seed)
    raw_model = MiniLlamaForCausalLM(config)
    raw_model.load_state_dict(source["model_state_dict"], strict=True)
    raw_model.to(ctx.device).eval()
    reference = MiniLlamaForCausalLM(config)
    reference.load_state_dict(source["model_state_dict"], strict=True)
    reference.to(ctx.device).eval().requires_grad_(False)
    optimizer = single.make_optimizer(raw_model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: single.lr_factor(index, total_updates, warmup_steps,
                                                  args.min_lr_ratio))
    progress = {"step": 0, "optimizer_updates": 0, "questions_seen": 0,
                "correct_samples": 0, "mixed_groups": 0}
    if checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        progress = checkpoint["progress"]
        if scheduler.last_epoch != progress["optimizer_updates"] or \
                progress["optimizer_updates"] != progress["step"] * args.policy_epochs or \
                not 0 <= progress["step"] < total_steps:
            raise ValueError("GRPO 恢复进度与学习率调度器不一致")
    model = (DDP(raw_model, device_ids=[ctx.device.index] if ctx.device.type == "cuda" else None,
                 broadcast_buffers=False) if ctx.distributed else raw_model)
    stop_step = min(total_steps, args.stop_after_steps or total_steps)
    if stop_step <= progress["step"]:
        raise ValueError("stop-after-steps 必须大于已有进度")
    output = (args.resume.parent if args.resume else args.output_dir).resolve()
    if not args.resume:
        ddp.rank_zero_call(ctx, lambda: output.mkdir(parents=True, exist_ok=False))
    writer = SummaryWriter(str(output / "tensorboard")) if ctx.rank == 0 else None
    try:
        single.restore_rng((checkpoint or source)["rng_states"][ctx.rank], ctx.device)
        started = time.perf_counter()
        cached_epoch = -1
        order = None
        for step in range(progress["step"], stop_step):
            epoch, offset = divmod(step, steps_per_epoch)
            if epoch != cached_epoch:
                order = torch.randperm(len(datasets["train"]),
                                       generator=torch.Generator().manual_seed(args.seed + epoch))
                cached_epoch = epoch
            indices = order[offset * global_batch:(offset + 1) * global_batch]
            batches, counts = make_rollout(raw_model, reference, datasets["train"],
                                           indices, tokenizer, config, args, ctx, epoch)
            for policy_epoch in range(args.policy_epochs):
                learning_rate = optimizer.param_groups[0]["lr"]
                metrics = update_policy(model, optimizer, batches, args, ctx)
                scheduler.step()
                progress["optimizer_updates"] += 1
                if writer:
                    for name, value in (*metrics.items(), ("learning_rate", learning_rate)):
                        writer.add_scalar(f"train/{name}", value, progress["optimizer_updates"])
            progress["step"] = step + 1
            progress["questions_seen"] += int(counts[0].item())
            progress["correct_samples"] += int(counts[1].item())
            progress["mixed_groups"] += int(counts[2].item())
            if writer:
                for name, value in (("correct_samples", counts[1].item()),
                                    ("mixed_groups", counts[2].item()),
                                    ("parsed_samples", counts[3].item())):
                    writer.add_scalar(f"rollout/{name}", value, step + 1)
                if (step + 1) % args.log_every == 0 or step == 0:
                    print(f"step={step + 1}/{total_steps} correct={int(counts[1])} "
                          f"mixed={int(counts[2])} kl={metrics['kl']:.5f} "
                          f"clip={metrics['clip_fraction']:.4f} "
                          f"grad_norm={metrics['gradient_norm']:.4f}", flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == stop_step:
                save(output / "checkpoint.pt", model, optimizer, scheduler, progress,
                     contract, tokenizer_path, ctx)
        summary = {"status": "complete" if progress["step"] == total_steps else "paused",
                   "model_parameters": raw_model.num_parameters(), "progress": progress,
                   "contract": contract, "audit": audit,
                   "dropped_epoch_tail": len(datasets["train"]) % global_batch,
                   "session_elapsed_seconds": round(time.perf_counter() - started, 2),
                   "checkpoint": str(output / "checkpoint.pt")}

        def write_summary():
            temporary = output / "summary.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
            temporary.replace(output / "summary.json")
            print(f"{summary['status']}: {output / 'checkpoint.pt'}", flush=True)
        ddp.rank_zero_call(ctx, write_summary)
    finally:
        if writer:
            writer.close()


def main():
    args = parser()
    torch.set_num_threads(args.cpu_threads)
    if args.audit_only:
        run(args, ddp.Context(0, 2, torch.device("cpu")))
        return
    ctx = ddp.setup_distributed(args.device)
    try:
        run(args, ctx)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
