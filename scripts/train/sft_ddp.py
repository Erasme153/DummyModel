#!/usr/bin/env python3
"""M9: assistant-only SFT on UltraFeedback Binarized or GSM8K."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time
import unicodedata

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
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


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("ultrafeedback", "gsm8k"),
                        default="ultrafeedback")
    result.add_argument("--raw-data-dir", type=Path)
    result.add_argument("--ultrafeedback-dir", type=Path,
                        default=ROOT / "data/raw/m09_ultrafeedback/data")
    result.add_argument("--holdout-size", type=int, default=256)
    result.add_argument("--split-max-new-tokens", type=int, default=256)
    result.add_argument("--init-checkpoint", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--resume", type=Path)
    result.add_argument("--audit-only", action="store_true")
    result.add_argument("--device", default="cuda")
    result.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--max-length", type=int, default=2048)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--grad-accum-steps", type=int, default=2)
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=2e-5)
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
    if args.raw_data_dir is None:
        args.raw_data_dir = ROOT / ("data/raw/m09_gsm8k/main" if args.dataset == "gsm8k"
                                    else "data/raw/m09_ultrafeedback/data")
    if args.init_checkpoint is None:
        args.init_checkpoint = ROOT / ("runs/m09_213m_sft/checkpoint.pt" if args.dataset == "gsm8k"
                                       else "runs/m08_213m_cooldown/checkpoint.pt")
    for name in ("max_length", "batch_size", "grad_accum_steps", "epochs", "eval_every",
                 "save_every", "log_every", "cpu_threads", "holdout_size",
                 "split_max_new_tokens"):
        if getattr(args, name) <= 0:
            result.error(f"{name} 必须为正数")
    if args.max_length < 2 or args.seed < 0:
        result.error("max-length 至少为 2，seed 不能为负数")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
        result.error("warmup-ratio/min-lr-ratio 超出范围")
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
           for name in ("learning_rate", "max_grad_norm")):
        result.error("learning-rate/max-grad-norm 必须为有限正数")
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


def normalized_prompt(value: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(value.split())).casefold()


def parquet_path(root: Path, split: str) -> Path:
    matches = list(root.glob(f"{split}-*.parquet"))
    if len(matches) != 1:
        raise ValueError(f"需要恰好一个 {split} Parquet 文件：{root}")
    return matches[0]


def load_examples(root: Path, tokenizer: Tokenizer, config: MiniLlamaConfig,
                  max_length: int):
    paths = {name: parquet_path(root, name)
             for name in ("train_sft", "test_sft", "test_prefs")}
    test_sft_ids = set(pq.read_table(paths["test_sft"], columns=["prompt_id"])
                       ["prompt_id"].to_pylist())
    test_prompts = set()
    for name in ("test_sft", "test_prefs"):
        test_prompts.update(normalized_prompt(value) for value in
                            pq.read_table(paths[name], columns=["prompt"])["prompt"].to_pylist())
    bos, eos = config.bos_token_id, config.eos_token_id
    if bos != tokenizer.token_to_id("<s>") or eos != tokenizer.token_to_id("</s>"):
        raise ValueError("checkpoint 与 tokenizer 的 BOS/EOS 不一致")
    datasets = {}
    stats = {}
    for split in ("train_sft", "test_sft"):
        seen = set()
        counts = {"raw": 0, "test_overlap": 0, "duplicate_prompt": 0,
                  "overlength": 0, "empty_answer": 0}
        examples = []
        for batch in pq.ParquetFile(paths[split]).iter_batches(
                batch_size=256, columns=["prompt", "prompt_id", "chosen"]):
            selected = []
            for row in batch.to_pylist():
                counts["raw"] += 1
                key = normalized_prompt(row["prompt"])
                if split == "train_sft" and key in test_prompts:
                    counts["test_overlap"] += 1
                    continue
                if key in seen:
                    counts["duplicate_prompt"] += 1
                    continue
                chosen = row["chosen"]
                if (not chosen or len(chosen) != 2 or chosen[0]["role"] != "user"
                        or chosen[1]["role"] != "assistant"
                        or chosen[0]["content"] != row["prompt"]
                        or not chosen[1]["content"].strip()):
                    counts["empty_answer"] += 1
                    continue
                selected.append((key, row["prompt"], chosen[1]["content"]))
            # 分段编码并在 token 边界拼接，首个回复 token 的标签不会误落到 prompt。
            prefixes = tokenizer.encode_batch(
                [f"[INST] {prompt} [/INST]" for _, prompt, _ in selected],
                add_special_tokens=False)
            completions = tokenizer.encode_batch(
                [" " + answer for _, _, answer in selected], add_special_tokens=False)
            for (key, _, _), prefix, completion in zip(selected, prefixes, completions):
                if key in seen:
                    counts["duplicate_prompt"] += 1
                    continue
                ids = (bos, *prefix.ids, *completion.ids, eos)
                if len(ids) > max_length or not completion.ids:
                    counts["overlength"] += 1
                    continue
                labels = (-100,) * (1 + len(prefix.ids)) + (*completion.ids, eos)
                seen.add(key)
                examples.append((ids, labels, len(completion.ids) + 1))
        counts["kept"] = len(examples)
        counts["supervised_tokens"] = sum(item[2] for item in examples)
        counts["input_tokens"] = sum(len(item[0]) for item in examples)
        datasets[split] = examples
        stats[split] = counts
    stats["source_sha256"] = {name: single.file_sha256(path) for name, path in paths.items()}
    stats["test_sft_subset_of_test_prefs"] = test_sft_ids.issubset(set(
        pq.read_table(paths["test_prefs"], columns=["prompt_id"])["prompt_id"].to_pylist()))
    if not stats["test_sft_subset_of_test_prefs"] or not datasets["train_sft"] or not datasets["test_sft"]:
        raise ValueError("SFT 训练/验证为空或测试划分不符合预期")
    return datasets, stats


def load_gsm8k_examples(root: Path, ultrafeedback_dir: Path, tokenizer: Tokenizer,
                        config: MiniLlamaConfig, max_length: int, *,
                        holdout_size: int = 256, split_max_new_tokens: int = 256):
    # 复用基线的过滤和 randperm 划分；train_only 只投影官方测试集 question 列。
    from scripts.eval import grpo_baseline

    split, audit = grpo_baseline.load_data(
        root, ultrafeedback_dir, tokenizer, config, holdout_size=holdout_size,
        max_new_tokens=split_max_new_tokens, seed=2026, train_only=True)
    datasets = {}
    for source, target in (("train", "train_sft"), ("holdout", "test_sft")):
        examples = []
        for row in split[source]:
            prefix = row["prompt_ids"]
            completion = tokenizer.encode(" " + row["solution"], add_special_tokens=False).ids
            ids = (*prefix, *completion, config.eos_token_id)
            if len(ids) > max_length or not completion:
                raise ValueError(f"GSM8K {source} 第 {row['source_index']} 题超长或答案为空；"
                                 "不能静默改变基线划分")
            labels = (-100,) * len(prefix) + (*completion, config.eos_token_id)
            examples.append((ids, labels, len(completion) + 1))
        datasets[target] = examples
    audit["train_sft"] = {"kept": len(datasets["train_sft"]),
                          "supervised_tokens": sum(row[2] for row in datasets["train_sft"])}
    audit["test_sft"] = {"kept": len(datasets["test_sft"]),
                         "supervised_tokens": sum(row[2] for row in datasets["test_sft"])}
    return datasets, audit


def collate(examples, device, eos):
    length = max(len(ids) for ids, _, _ in examples)
    ids = torch.full((len(examples), length), eos, dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), length), dtype=torch.bool)
    for row, (tokens, targets, _) in enumerate(examples):
        size = len(tokens)
        ids[row, :size] = torch.tensor(tokens, dtype=torch.long)
        labels[row, :size] = torch.tensor(targets, dtype=torch.long)
        mask[row, :size] = True
    return ids.to(device), labels.to(device), mask.to(device)


@torch.inference_mode()
def evaluate(model, examples, args, ctx, eos):
    was_training = model.training
    model.eval()
    raw_model = ddp.unwrap(model)
    start = len(examples) * ctx.rank // ctx.world_size
    end = len(examples) * (ctx.rank + 1) // ctx.world_size
    total = 0.0
    count = 0
    try:
        for cursor in range(start, end, args.batch_size):
            part = examples[cursor:min(cursor + args.batch_size, end)]
            ids, labels, mask = collate(part, ctx.device, eos)
            with single.precision_context(ctx.device, args.precision):
                loss = raw_model(ids, labels=labels, attention_mask=mask).loss
            supervised = sum(example[2] for example in part)
            total += loss.item() * supervised
            count += supervised
        sums = torch.tensor([total, count], dtype=torch.float64, device=ctx.device)
        if ctx.distributed:
            dist.all_reduce(sums)
        if not torch.isfinite(sums).all() or sums[1] <= 0:
            raise RuntimeError("验证 loss 或监督 token 数无效")
        return (sums[0] / sums[1]).item(), int(sums[1].item())
    finally:
        model.train(was_training)


def train_step(model, optimizer, examples, indices, args, ctx, eos):
    optimizer.zero_grad(set_to_none=True)
    local_count = args.batch_size * args.grad_accum_steps
    global_tokens = sum(examples[int(index)][2] for index in indices)
    local = indices[ctx.rank * local_count:(ctx.rank + 1) * local_count]
    loss_sum = 0.0
    for micro in range(args.grad_accum_steps):
        part = [examples[int(index)] for index in
                local[micro * args.batch_size:(micro + 1) * args.batch_size]]
        ids, labels, mask = collate(part, ctx.device, eos)
        supervised = sum(example[2] for example in part)
        sync = nullcontext() if micro == args.grad_accum_steps - 1 or not ctx.distributed else model.no_sync()
        with sync:
            with single.precision_context(ctx.device, args.precision):
                loss = model(ids, labels=labels, attention_mask=mask).loss
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError("SFT loss 非有限")
            (loss * (supervised * ctx.world_size / global_tokens)).backward()
        loss_sum += loss.detach().item() * supervised
    norm = clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    sums = torch.tensor([loss_sum, 0.0], dtype=torch.float64, device=ctx.device)
    if ctx.distributed:
        dist.all_reduce(sums)
    return sums[0].item() / global_tokens, global_tokens, float(norm)


def save(path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx):
    rng = single.rng_state(ctx.device)
    states = [None] * ctx.world_size
    if ctx.distributed:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng

    def write():
        value = {"format_version": 3, "model_config": ddp.unwrap(model).config.to_dict(),
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
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    dataset = getattr(args, "dataset", "ultrafeedback")
    expected_version = 3 if dataset == "gsm8k" else 2
    if source.get("format_version") != expected_version or source.get("contract", {}).get("world_size") != ctx.world_size:
        raise ValueError(f"{dataset} SFT 起点需要同卡数的 v{expected_version} checkpoint")
    config = MiniLlamaConfig.from_dict(source["model_config"])
    if config.num_experts:
        raise ValueError("当前 SFT 入口只支持 dense 模型")
    if args.max_length > config.max_position_embeddings:
        raise ValueError("max-length 超过模型上下文长度")
    tokenizer_path = Path(source["tokenizer_path"])
    tokenizer_sha = single.file_sha256(tokenizer_path)
    if tokenizer_sha != source["contract"]["data"]["tokenizer_sha256"]:
        raise ValueError("SFT 起点 tokenizer 指纹不符")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("SFT tokenizer 与模型词表不符")
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    if dataset == "gsm8k":
        datasets, audit = load_gsm8k_examples(
            args.raw_data_dir, args.ultrafeedback_dir, tokenizer, config,
            args.max_length, holdout_size=args.holdout_size,
            split_max_new_tokens=args.split_max_new_tokens)
    else:
        datasets, audit = load_examples(args.raw_data_dir, tokenizer, config, args.max_length)
    global_batch = args.batch_size * args.grad_accum_steps * ctx.world_size
    steps_per_epoch = len(datasets["train_sft"]) // global_batch
    if steps_per_epoch == 0:
        raise ValueError("训练样本不足一个全局 batch")
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    contract = {"model_config": config.to_dict(),
                "data": {"tokenizer_sha256": tokenizer_sha, "sequence_length": args.max_length,
                         "source_sha256": audit["source_sha256"],
                         "chat_template": "bos+[INST] {prompt} [/INST]+assistant+eos; NFKC-space-casefold dedup"},
                "recipe": {name: getattr(args, name) for name in
                           ("batch_size", "grad_accum_steps", "epochs", "learning_rate",
                            "min_lr_ratio", "warmup_ratio", "weight_decay", "beta1", "beta2",
                            "max_grad_norm", "seed", "precision")},
                "world_size": ctx.world_size, "global_batch": global_batch,
                "total_steps": total_steps, "source_checkpoint": str(args.init_checkpoint.resolve())}
    if dataset == "gsm8k":
        contract["data"].update(dataset="gsm8k", split_seed=2026,
                                holdout_size=args.holdout_size,
                                split_max_new_tokens=args.split_max_new_tokens,
                                holdout_source_indices_sha256=audit["holdout_source_indices_sha256"],
                                test_questions_sha256=audit["test_questions_sha256"],
                                chat_template="bos+[INST] {question}\\nSolve the problem. End your answer with #### <number>. [/INST]+assistant+eos")
    if args.audit_only:
        if ctx.rank == 0:
            print(json.dumps({"audit": audit, "global_batch": global_batch,
                              "steps_per_epoch": steps_per_epoch,
                              "dropped_epoch_tail": len(datasets["train_sft"]) % global_batch},
                             indent=2, ensure_ascii=False))
        return
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint and (checkpoint.get("format_version") != 3 or checkpoint.get("contract") != contract):
        raise ValueError("SFT 恢复约定与 checkpoint 不一致")
    torch.manual_seed(args.seed)
    raw_model = MiniLlamaForCausalLM(config)
    raw_model.load_state_dict(source["model_state_dict"], strict=True)
    raw_model.to(ctx.device)
    optimizer = single.make_optimizer(raw_model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: single.lr_factor(index, total_steps, warmup_steps,
                                                  args.min_lr_ratio))
    progress = {"step": 0, "input_tokens_seen": 0, "supervised_tokens_seen": 0,
                "initial_validation_loss": None, "validation_loss": None,
                "validation_supervised_tokens": None}
    if checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        progress = checkpoint["progress"]
        if scheduler.last_epoch != progress["step"] or not 0 <= progress["step"] < total_steps:
            raise ValueError("SFT 恢复进度与学习率调度器不一致")
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
            validation, val_tokens = evaluate(model, datasets["test_sft"], args, ctx,
                                              config.eos_token_id)
            progress.update(initial_validation_loss=validation, validation_loss=validation,
                            validation_supervised_tokens=val_tokens)
            if writer:
                writer.add_scalar("validation/loss", validation, 0)
        started = time.perf_counter()
        cached_epoch = -1
        order = None
        for step in range(progress["step"], stop_step):
            epoch, offset = divmod(step, steps_per_epoch)
            if epoch != cached_epoch:
                order = torch.randperm(len(datasets["train_sft"]),
                                       generator=torch.Generator().manual_seed(args.seed + epoch))
                cached_epoch = epoch
            indices = order[offset * global_batch:(offset + 1) * global_batch]
            learning_rate = optimizer.param_groups[0]["lr"]
            loss, supervised, grad_norm = train_step(
                model, optimizer, datasets["train_sft"], indices, args, ctx, config.eos_token_id)
            scheduler.step()
            progress["step"] = step + 1
            progress["supervised_tokens_seen"] += supervised
            progress["input_tokens_seen"] += sum(len(datasets["train_sft"][int(i)][0]) for i in indices)
            if writer:
                for name, value in (("loss", loss), ("learning_rate", learning_rate),
                                    ("gradient_norm", grad_norm),
                                    ("supervised_tokens_seen", progress["supervised_tokens_seen"])):
                    writer.add_scalar(f"train/{name}", value, step + 1)
                if (step + 1) % args.log_every == 0 or step == 0:
                    print(f"step={step + 1}/{total_steps} loss={loss:.6f} "
                          f"lr={learning_rate:.3e} grad_norm={grad_norm:.4f}", flush=True)
            if (step + 1) % args.eval_every == 0 or step + 1 == stop_step:
                validation, val_tokens = evaluate(model, datasets["test_sft"], args, ctx,
                                                  config.eos_token_id)
                progress.update(validation_loss=validation, validation_supervised_tokens=val_tokens)
                if writer:
                    writer.add_scalar("validation/loss", validation, step + 1)
                    writer.flush()
            if (step + 1) % args.save_every == 0 or step + 1 == stop_step:
                save(output / "checkpoint.pt", model, optimizer, scheduler, progress,
                     contract, tokenizer_path, ctx)
        summary = {"status": "complete" if progress["step"] == total_steps else "paused",
                   "model_parameters": raw_model.num_parameters(), "progress": progress,
                   "contract": contract, "audit": audit,
                   "dropped_epoch_tail": len(datasets["train_sft"]) % global_batch,
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
        # 数据审计不需要初始化 CUDA/DDP，按计划的双卡数计算完整训练步数。
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
