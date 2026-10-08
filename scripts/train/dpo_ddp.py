#!/usr/bin/env python3
"""M9: DPO from an SFT checkpoint on UltraFeedback Binarized preferences."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
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
from scripts.train import pretrain as single  # noqa: E402
from scripts.train import pretrain_ddp as ddp  # noqa: E402
from scripts.train import sft_ddp as sft  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--raw-data-dir", type=Path,
                        default=ROOT / "data/raw/m09_ultrafeedback/data")
    result.add_argument("--init-checkpoint", type=Path,
                        default=ROOT / "runs/m09_213m_sft/checkpoint.pt")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--resume", type=Path)
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--max-length", type=int, default=2048)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--grad-accum-steps", type=int, default=4)
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=5e-6)
    result.add_argument("--beta", type=float, default=0.1)
    result.add_argument("--min-lr-ratio", type=float, default=0.1)
    result.add_argument("--warmup-ratio", type=float, default=0.03)
    result.add_argument("--weight-decay", type=float, default=0.1)
    result.add_argument("--beta1", type=float, default=0.9)
    result.add_argument("--beta2", type=float, default=0.95)
    result.add_argument("--max-grad-norm", type=float, default=1.0)
    result.add_argument("--eval-every", type=int, default=500)
    result.add_argument("--save-every", type=int, default=1000)
    result.add_argument("--log-every", type=int, default=50)
    result.add_argument("--stop-after-steps", type=int)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--cpu-threads", type=int, default=4)
    args = result.parse_args()
    for name in ("max_length", "batch_size", "grad_accum_steps", "epochs", "eval_every",
                 "save_every", "log_every", "cpu_threads"):
        if getattr(args, name) <= 0:
            result.error(f"{name} 必须为正数")
    if args.max_length < 2 or args.seed < 0:
        result.error("max-length 至少为 2，seed 不能为负数")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
        result.error("warmup-ratio/min-lr-ratio 超出范围")
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
           for name in ("learning_rate", "beta", "max_grad_norm")):
        result.error("learning-rate/beta/max-grad-norm 必须为有限正数")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        result.error("weight-decay 必须为有限非负数")
    if not all(0 <= value < 1 for value in (args.beta1, args.beta2)):
        result.error("beta1/beta2 必须在 [0,1) 内")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        result.error("stop-after-steps 必须为正数")
    if args.resume and args.output_dir:
        result.error("--resume 使用原输出目录，不再指定 --output-dir")
    if not args.audit_only and not args.resume and not args.output_dir:
        result.error("新训练必须指定 --output-dir")
    return args


def load_examples(root: Path, tokenizer: Tokenizer, config: MiniLlamaConfig,
                  max_length: int):
    paths = {name: sft.parquet_path(root, name)
             for name in ("train_prefs", "test_prefs", "test_sft")}
    sft_test = pq.read_table(paths["test_sft"], columns=["prompt", "prompt_id"])
    sft_ids = set(sft_test["prompt_id"].to_pylist())
    sft_keys = {sft.normalized_prompt(value) for value in sft_test["prompt"].to_pylist()}
    prefs_test = pq.read_table(paths["test_prefs"], columns=["prompt"])
    prefs_test_keys = {sft.normalized_prompt(value)
                       for value in prefs_test["prompt"].to_pylist()}
    if config.bos_token_id != tokenizer.token_to_id("<s>") or \
            config.eos_token_id != tokenizer.token_to_id("</s>"):
        raise ValueError("checkpoint 与 tokenizer 的 BOS/EOS 不一致")

    datasets, audit = {}, {}
    for split in ("train_prefs", "test_prefs"):
        counts = {name: 0 for name in
                  ("raw", "train_test_overlap", "sft_validation_overlap", "duplicate_prompt",
                   "tie_or_reverse", "invalid_or_identical", "overlength_or_empty")}
        seen, examples = set(), []
        for batch in pq.ParquetFile(paths[split]).iter_batches(
                batch_size=256, columns=["prompt", "prompt_id", "chosen", "rejected",
                                         "score_chosen", "score_rejected"]):
            selected = []
            for row in batch.to_pylist():
                counts["raw"] += 1
                key = sft.normalized_prompt(row["prompt"])
                if split == "train_prefs" and key in prefs_test_keys:
                    counts["train_test_overlap"] += 1
                    continue
                if split == "test_prefs" and (key in sft_keys or row["prompt_id"] in sft_ids):
                    counts["sft_validation_overlap"] += 1
                    continue
                if key in seen:
                    counts["duplicate_prompt"] += 1
                    continue
                chosen_score, rejected_score = row["score_chosen"], row["score_rejected"]
                if (chosen_score is None or rejected_score is None or
                        not math.isfinite(chosen_score) or not math.isfinite(rejected_score) or
                        chosen_score <= rejected_score):
                    counts["tie_or_reverse"] += 1
                    continue
                chosen, rejected = row["chosen"], row["rejected"]
                if (any(not pair or len(pair) != 2 or pair[0]["role"] != "user" or
                        pair[1]["role"] != "assistant" or
                        pair[0]["content"] != row["prompt"] or
                        not pair[1]["content"].strip() for pair in (chosen, rejected)) or
                        sft.normalized_prompt(chosen[1]["content"]) ==
                        sft.normalized_prompt(rejected[1]["content"])):
                    counts["invalid_or_identical"] += 1
                    continue
                selected.append((key, row["prompt"], chosen[1]["content"],
                                 rejected[1]["content"]))
            prefixes = tokenizer.encode_batch(
                [f"[INST] {prompt} [/INST]" for _, prompt, _, _ in selected],
                add_special_tokens=False)
            chosen_tokens = tokenizer.encode_batch(
                [" " + answer for _, _, answer, _ in selected], add_special_tokens=False)
            rejected_tokens = tokenizer.encode_batch(
                [" " + answer for _, _, _, answer in selected], add_special_tokens=False)
            for (key, _, _, _), prefix, chosen, rejected in zip(
                    selected, prefixes, chosen_tokens, rejected_tokens):
                if key in seen:
                    counts["duplicate_prompt"] += 1
                    continue
                prefix_ids = (config.bos_token_id, *prefix.ids)
                pair = []
                for completion in (chosen.ids, rejected.ids):
                    ids = (*prefix_ids, *completion, config.eos_token_id)
                    labels = (-100,) * len(prefix_ids) + (*completion, config.eos_token_id)
                    pair.append((ids, labels, len(completion) + 1))
                if (not chosen.ids or not rejected.ids or
                        any(len(item[0]) > max_length for item in pair)):
                    counts["overlength_or_empty"] += 1
                    continue
                seen.add(key)
                examples.append(tuple(pair))
        counts["kept"] = len(examples)
        counts["chosen_tokens"] = sum(pair[0][2] for pair in examples)
        counts["rejected_tokens"] = sum(pair[1][2] for pair in examples)
        datasets[split] = examples
        audit[split] = counts
    audit["source_sha256"] = {name: single.file_sha256(path) for name, path in paths.items()}
    audit["test_sft_subset_of_test_prefs"] = sft_ids.issubset(set(
        pq.read_table(paths["test_prefs"], columns=["prompt_id"])["prompt_id"].to_pylist()))
    if not audit["test_sft_subset_of_test_prefs"] or not all(datasets.values()):
        raise ValueError("偏好训练/独立评测为空，或 SFT 验证集划分不符合预期")
    return datasets, audit


def collate(pairs, device, eos):
    # 前半为 chosen，后半为 rejected；两个回答共用同一条 prompt。
    return sft.collate([pair[0] for pair in pairs] + [pair[1] for pair in pairs],
                       device, eos)


def sequence_logps(logits, labels):
    """仅将回答与 EOS 的 next-token log-prob 求和，每条序列返回一个值。"""
    targets = labels[:, 1:]
    token_losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1), ignore_index=-100, reduction="none")
    return -token_losses.reshape(targets.shape).sum(dim=1)


def dpo_objective(policy_logps, reference_logps, beta):
    if policy_logps.shape != reference_logps.shape or policy_logps.ndim != 1 or \
            policy_logps.numel() % 2:
        raise ValueError("DPO log-prob 必须是按 chosen、rejected 排列的等长一维向量")
    count = policy_logps.numel() // 2
    chosen_reward = beta * (policy_logps[:count] - reference_logps[:count])
    rejected_reward = beta * (policy_logps[count:] - reference_logps[count:])
    margin = chosen_reward - rejected_reward
    return F.softplus(-margin), margin, (policy_logps[:count] > policy_logps[count:])


def batch_metrics(policy, reference, ids, labels, mask, precision, device, beta):
    with single.precision_context(device, precision):
        logits = policy(ids, attention_mask=mask).logits
    policy_logps = sequence_logps(logits, labels)
    with torch.no_grad():
        with single.precision_context(device, precision):
            reference_logits = reference(ids, attention_mask=mask).logits
        reference_logps = sequence_logps(reference_logits, labels)
    return dpo_objective(policy_logps, reference_logps, beta)


@torch.inference_mode()
def evaluate(model, reference, examples, args, ctx, eos):
    was_training = model.training
    model.eval()
    start = len(examples) * ctx.rank // ctx.world_size
    end = len(examples) * (ctx.rank + 1) // ctx.world_size
    totals = torch.zeros(5, dtype=torch.float64, device=ctx.device)
    try:
        for cursor in range(start, end, args.batch_size):
            part = examples[cursor:min(cursor + args.batch_size, end)]
            ids, labels, mask = collate(part, ctx.device, eos)
            losses, margins, policy_wins = batch_metrics(
                ddp.unwrap(model), reference, ids, labels, mask,
                args.precision, ctx.device, args.beta)
            totals += torch.tensor([losses.sum().item(), margins.sum().item(),
                                    (margins > 0).sum().item(), policy_wins.sum().item(),
                                    len(part)], dtype=torch.float64, device=ctx.device)
        if ctx.distributed:
            dist.all_reduce(totals)
        if not torch.isfinite(totals).all() or totals[-1] != len(examples):
            raise RuntimeError("DPO 验证数值或样本数无效")
        return {"loss": (totals[0] / totals[-1]).item(),
                "reward_margin": (totals[1] / totals[-1]).item(),
                "reward_accuracy": (totals[2] / totals[-1]).item(),
                "policy_preference_accuracy": (totals[3] / totals[-1]).item(),
                "pairs": int(totals[-1].item())}
    finally:
        model.train(was_training)


def train_step(model, reference, optimizer, examples, indices, args, ctx, eos):
    optimizer.zero_grad(set_to_none=True)
    local_count = args.batch_size * args.grad_accum_steps
    local = indices[ctx.rank * local_count:(ctx.rank + 1) * local_count]
    totals = torch.zeros(4, dtype=torch.float64, device=ctx.device)
    for micro in range(args.grad_accum_steps):
        part = [examples[int(index)] for index in
                local[micro * args.batch_size:(micro + 1) * args.batch_size]]
        ids, labels, mask = collate(part, ctx.device, eos)
        sync = nullcontext() if micro == args.grad_accum_steps - 1 or not ctx.distributed else model.no_sync()
        with sync:
            losses, margins, policy_wins = batch_metrics(
                model, reference, ids, labels, mask,
                args.precision, ctx.device, args.beta)
            if not torch.isfinite(losses).all():
                raise RuntimeError("DPO loss 非有限")
            (losses.sum() * ctx.world_size / len(indices)).backward()
        totals += torch.tensor([losses.detach().sum().item(), margins.detach().sum().item(),
                                (margins.detach() > 0).sum().item(),
                                policy_wins.sum().item()],
                               dtype=torch.float64, device=ctx.device)
    norm = clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if ctx.distributed:
        dist.all_reduce(totals)
    return {"loss": (totals[0] / len(indices)).item(),
            "reward_margin": (totals[1] / len(indices)).item(),
            "reward_accuracy": (totals[2] / len(indices)).item(),
            "policy_preference_accuracy": (totals[3] / len(indices)).item(),
            "gradient_norm": float(norm)}


def save(path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx):
    rng = single.rng_state(ctx.device)
    states = [None] * ctx.world_size
    if ctx.distributed:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng

    def write():
        value = {"format_version": 5, "model_config": ddp.unwrap(model).config.to_dict(),
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
    if source.get("format_version") != 3 or source.get("contract", {}).get("world_size") != ctx.world_size:
        raise ValueError("DPO 起点需要同卡数的 SFT v3 checkpoint")
    config = MiniLlamaConfig.from_dict(source["model_config"])
    if config.num_experts:
        raise ValueError("当前 DPO 入口只支持 dense 模型")
    if args.max_length > config.max_position_embeddings:
        raise ValueError("max-length 超过模型上下文长度")
    tokenizer_path = Path(source["tokenizer_path"])
    tokenizer_sha = single.file_sha256(tokenizer_path)
    if tokenizer_sha != source["contract"]["data"]["tokenizer_sha256"]:
        raise ValueError("DPO 起点 tokenizer 指纹不符")
    source_sha = single.file_sha256(args.init_checkpoint)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("DPO tokenizer 与模型词表不符")
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    datasets, audit = load_examples(args.raw_data_dir, tokenizer, config, args.max_length)
    global_batch = args.batch_size * args.grad_accum_steps * ctx.world_size
    steps_per_epoch = len(datasets["train_prefs"]) // global_batch
    if not steps_per_epoch:
        raise ValueError("训练配对不足一个全局 batch")
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    contract = {"model_config": config.to_dict(),
                "data": {"tokenizer_sha256": tokenizer_sha, "sequence_length": args.max_length,
                         "source_sha256": audit["source_sha256"],
                         "chat_template": "bos+[INST] {prompt} [/INST]+assistant+eos; NFKC-space-casefold dedup"},
                "recipe": {name: getattr(args, name) for name in
                           ("batch_size", "grad_accum_steps", "epochs", "learning_rate",
                            "beta", "min_lr_ratio", "warmup_ratio", "weight_decay", "beta1",
                            "beta2", "max_grad_norm", "seed", "precision")},
                "world_size": ctx.world_size, "global_batch": global_batch,
                "total_steps": total_steps,
                "source_checkpoint": str(args.init_checkpoint.resolve()),
                "source_checkpoint_sha256": source_sha,
                "reference_checkpoint": str(args.init_checkpoint.resolve())}
    if args.audit_only:
        if ctx.rank == 0:
            print(json.dumps({"audit": audit, "global_batch": global_batch,
                              "steps_per_epoch": steps_per_epoch,
                              "dropped_epoch_tail": len(datasets["train_prefs"]) % global_batch},
                             indent=2, ensure_ascii=False))
        return
    checkpoint = (torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
                  if args.resume else None)
    if checkpoint and (checkpoint.get("format_version") != 5 or
                       checkpoint.get("contract") != contract):
        raise ValueError("DPO 恢复约定与 checkpoint 不一致")
    torch.manual_seed(args.seed)
    raw_model = MiniLlamaForCausalLM(config)
    raw_model.load_state_dict(source["model_state_dict"], strict=True)
    raw_model.to(ctx.device)
    reference = MiniLlamaForCausalLM(config)
    reference.load_state_dict(source["model_state_dict"], strict=True)
    reference.to(ctx.device).eval()
    reference.requires_grad_(False)
    optimizer = single.make_optimizer(raw_model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: single.lr_factor(index, total_steps, warmup_steps,
                                                  args.min_lr_ratio))
    progress = {"step": 0, "pairs_seen": 0, "input_tokens_seen": 0,
                "initial_validation": None, "validation": None}
    if checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        progress = checkpoint["progress"]
        if scheduler.last_epoch != progress["step"] or not 0 <= progress["step"] < total_steps:
            raise ValueError("DPO 恢复进度与学习率调度器不一致")
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
        if checkpoint:
            single.restore_rng(checkpoint["rng_states"][ctx.rank], ctx.device)
        else:
            single.restore_rng(source["rng_states"][ctx.rank], ctx.device)
        if progress["step"] == 0:
            metrics = evaluate(model, reference, datasets["test_prefs"], args, ctx,
                               config.eos_token_id)
            progress.update(initial_validation=metrics, validation=metrics)
            if writer:
                for name, value in metrics.items():
                    writer.add_scalar(f"validation/{name}", value, 0)
        started = time.perf_counter()
        cached_epoch = -1
        order = None
        for step in range(progress["step"], stop_step):
            epoch, offset = divmod(step, steps_per_epoch)
            if epoch != cached_epoch:
                order = torch.randperm(len(datasets["train_prefs"]),
                                       generator=torch.Generator().manual_seed(args.seed + epoch))
                cached_epoch = epoch
            indices = order[offset * global_batch:(offset + 1) * global_batch]
            learning_rate = optimizer.param_groups[0]["lr"]
            metrics = train_step(model, reference, optimizer, datasets["train_prefs"],
                                 indices, args, ctx, config.eos_token_id)
            scheduler.step()
            progress["step"] = step + 1
            progress["pairs_seen"] += global_batch
            progress["input_tokens_seen"] += sum(
                len(ids) for index in indices for ids, _, _ in
                datasets["train_prefs"][int(index)])
            if writer:
                for name, value in (*metrics.items(), ("learning_rate", learning_rate)):
                    writer.add_scalar(f"train/{name}", value, step + 1)
                if (step + 1) % args.log_every == 0 or step == 0:
                    print(f"step={step + 1}/{total_steps} loss={metrics['loss']:.6f} "
                          f"margin={metrics['reward_margin']:.4f} "
                          f"lr={learning_rate:.3e} grad_norm={metrics['gradient_norm']:.4f}",
                          flush=True)
            if (step + 1) % args.eval_every == 0 or step + 1 == stop_step:
                metrics = evaluate(model, reference, datasets["test_prefs"], args, ctx,
                                   config.eos_token_id)
                progress["validation"] = metrics
                if writer:
                    for name, value in metrics.items():
                        writer.add_scalar(f"validation/{name}", value, step + 1)
                    writer.flush()
            if (step + 1) % args.save_every == 0 or step + 1 == stop_step:
                save(output / "checkpoint.pt", model, optimizer, scheduler, progress,
                     contract, tokenizer_path, ctx)
        summary = {"status": "complete" if progress["step"] == total_steps else "paused",
                   "model_parameters": raw_model.num_parameters(), "progress": progress,
                   "contract": contract, "audit": audit,
                   "dropped_epoch_tail": len(datasets["train_prefs"]) % global_batch,
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
        ctx = ddp.Context(0, 2, torch.device("cpu"))
        run(args, ctx)
        return
    ctx = ddp.setup_distributed(args.device)
    try:
        run(args, ctx)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
