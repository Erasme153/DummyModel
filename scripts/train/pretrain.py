#!/usr/bin/env python3
"""M1 单卡预训练：定长 token 数据 -> MiniLlama -> AdamW -> 验证与断点保存。

沿用 prepare_fineweb.py 生成的二进制数据，默认训练约 39M 模型。这里的 step
专指一次 optimizer.step()；一次 step 可以包含多个 micro-batch 的反向传播。
一个 epoch 会恰好遍历全部训练序列，最后不足一个有效 batch 的数据也会参与训练。

为了方便学习，数据读取、梯度累积、学习率和恢复过程都在本文件中显式实现。
不启动多进程 DataLoader：通过固定种子的排列和“下一条序列位置”恢复读取顺序，
避免预取队列让恢复位置难以理解。当前不支持多卡、FP16 或训练途中更换数据。

参考 PyTorch 官方 AMP、优化器与序列化文档，链接见 M1 实验 README。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=PROJECT_ROOT / "data/tokenized/m01_fineweb_100m")
    parser.add_argument("--model-config", type=Path,
                        default=PROJECT_ROOT / "configs/model/ladder/v001/p039m.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda",
                        help="cuda、cuda:N 或 cpu；N 是当前进程可见 GPU 的逻辑编号，从 0 开始")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4, help="每个 micro-batch 的序列数")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="0 验证全部数据；正数只验证固定前 N 个 batch，适用于快速调试")
    parser.add_argument("--stop-after-steps", type=int,
                        help="在指定总更新步数暂停并保存，不改变由 epochs 决定的完整 LR 计划")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    for name in ("batch_size", "grad_accum_steps", "epochs", "eval_every", "save_every",
                 "log_every", "cpu_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正数")
    for name in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} 必须是有限正数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay 必须是有限非负数")
    if not 0 <= args.min_lr_ratio <= 1 or not all(0 <= b < 1 for b in (args.beta1, args.beta2)):
        parser.error("min-lr-ratio 必须在 [0, 1]，AdamW betas 必须在 [0, 1)")
    if min(args.warmup_steps, args.eval_batches, args.seed) < 0:
        parser.error("warmup-steps、eval-batches、seed 不能为负数")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        parser.error("stop-after-steps 必须为正数")
    return args


def resolve_device(requested: str) -> torch.device:
    """检查当前进程的逻辑编号，而不是用宿主机的 GPU 编号直接寻址。

    例如 CUDA_VISIBLE_DEVICES=1 只暴露原来的第 1 号卡，但进程内部只有一张卡，
    编号变为 cuda:0。此时 --device cuda:1 就越界。CUDA_VISIBLE_DEVICES=1,0 时，
    则 cuda:0 对应原来的 1 号卡，cuda:1 对应原来的 0 号卡。

    显式指定越界编号时直接报错，不自动换卡，以免把任务放到其他正在使用的 GPU。
    """
    device = torch.device(requested)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("本脚本只支持单张 CUDA GPU 或 CPU")
    if device.type == "cpu":
        return device
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if count == 0 or not torch.cuda.is_available():
        raise RuntimeError(
            f"当前进程无法访问 CUDA：可见 GPU 数={count}，CUDA_VISIBLE_DEVICES={visible!r}。"
            "请检查运行该命令的 Python 环境和设备权限；None 表示未设置该环境变量。"
        )
    # cuda 不带序号时沿用当前设备；显式 cuda:N 则必须满足 0 <= N < 可见 GPU 数。
    index = torch.cuda.current_device() if device.index is None else device.index
    if not 0 <= index < count:
        raise ValueError(
            f"--device {requested} 超出范围：当前进程仅可见 {count} 张 GPU，"
            f"可用逻辑编号为 cuda:0 至 cuda:{count - 1}，"
            f"CUDA_VISIBLE_DEVICES={visible!r}。"
            "例如 CUDA_VISIBLE_DEVICES=1 时应使用 --device cuda:0，而非 cuda:1；"
            "修改可见性后应重新启动 Python/调试进程。"
        )
    return torch.device("cuda", index)


def file_sha256(path: Path) -> str:
    """分块计算指纹，恢复时发现“文件路径没变，但数据内容被替换”的情况。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PackedDataset:
    """只读映射准备好的 uint16 文件，实际取 batch 时才复制并转换成 int64。

    memmap 让操作系统按需读取数据，不要求启动时把整个训练集复制到 Python 堆。
    数据每行固定 T 个 token，没有 padding，因此每条序列都贡献 T-1 个 loss 目标。
    """

    def __init__(self, path: Path, stats: dict, sequence_length: int, vocab_size: int):
        tokens = stats["written_tokens"]
        if sequence_length < 2 or tokens <= 0 or tokens % sequence_length:
            raise ValueError(f"无效的序列长度或 token 数：{path}")
        if stats["sequences"] != tokens // sequence_length:
            raise ValueError(f"序列数与 token 数不一致：{path}")
        if path.stat().st_size != tokens * 2 or stats["bytes"] != tokens * 2:
            raise ValueError(f"文件字节数与摘要不一致：{path}")
        self.rows = np.memmap(path, dtype="<u2", mode="r").reshape(-1, sequence_length)
        # 启动时检查所有 ID，避免运行很久后才在 Embedding 遇到越界。uint16 本身
        # 不会有负数；真正要限制的是最大 ID，不能只检查数据类型。
        if self.rows.max() >= vocab_size:
            raise ValueError(f"存在超出词表的 token ID：{path}")

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, indices, device: torch.device) -> torch.Tensor:
        # astype 创建独立、可写的 int64 batch，再交给 PyTorch；不会修改只读原文件。
        # 不在这里 shift labels：模型内部已有 logits[:, :-1] -> labels[:, 1:]。
        array = self.rows[indices].astype(np.int64)
        return torch.from_numpy(array).to(device)


def load_data(directory: Path, config: MiniLlamaConfig):
    metadata = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or metadata["format"]["dtype"] != "<u2":
        raise ValueError("数据必须准备完成，且存储格式为小端 uint16")
    sequence_length = metadata["format"]["sequence_length"]
    if sequence_length > config.max_position_embeddings:
        raise ValueError("数据序列长度超过模型支持的最大上下文")
    tokenizer_path = directory / metadata["tokenizer"]["file"]
    tokenizer_sha = file_sha256(tokenizer_path)
    if tokenizer_sha != metadata["tokenizer"]["sha256"]:
        raise ValueError("tokenizer 内容与数据摘要的指纹不一致")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size or config.vocab_size != metadata["tokenizer"]["vocab_size"]:
        raise ValueError("模型、tokenizer 和数据摘要的词表大小不一致")
    if config.eos_token_id != metadata["tokenizer"]["eos_token_id"] or config.eos_token_id != tokenizer.token_to_id("</s>"):
        raise ValueError("模型与数据的 EOS 约定不一致")
    if config.bos_token_id != tokenizer.token_to_id("<s>"):
        raise ValueError("模型与 tokenizer 的 BOS ID 不一致")
    datasets = {}
    fingerprint = {"tokenizer_sha256": tokenizer_sha, "sequence_length": sequence_length}
    for split in ("train", "validation"):
        path = directory / f"{split}.bin"
        datasets[split] = PackedDataset(path, metadata["splits"][split], sequence_length, config.vocab_size)
        fingerprint[f"{split}_sha256"] = file_sha256(path)
    return datasets, fingerprint, tokenizer_path


def lr_factor(index: int, total_steps: int, warmup_steps: int, minimum: float) -> float:
    """返回第 index 次更新使用的 LR 倍率，index 从 0 开始。

    warmup=100 时，前 100 次更新从 peak/100 线性升到 peak；之后用半个余弦
    从 peak 降到 peak*minimum。total_steps 来自完整 epochs，与短跑暂停点无关。
    LambdaLR 初始化时就计算 index=0；每次 optimizer.step 后 scheduler.step，
    为下一次更新准备 LR。日志记录的是本次实际用过的值，而非准备好的下一次值。
    """
    index = min(index, total_steps - 1)
    if index < warmup_steps:
        return (index + 1) / warmup_steps
    progress = (index - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, args):
    # 二维及以上的矩阵使用 weight decay；RMSNorm 的一维缩放参数不衰减。
    # model.parameters() 会对共享的 Embedding / LM Head 去重，不会更新两次。
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(args.beta1, args.beta2))


def precision_context(device: torch.device, precision: str):
    # BF16 只包裹 forward/loss：矩阵运算可用 BF16，参数和 AdamW 状态仍保留 FP32。
    # backward 放在上下文之外，由对应 forward 的数据类型决定反向算子的精度。
    # BF16 的指数范围接近 FP32，本脚本不启用面向 FP16 下溢问题的 GradScaler。
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def train_update(model, optimizer, dataset, indices, args, device):
    """完成一个有效 batch 的梯度累积和一次更新，返回实际平均 loss 与裁剪前范数。

    例：micro-batch=4，累积 4 次，通常一次更新用 16 条序列。但若 epoch 尾部
    只剩 6 条，则两次 forward 分别用 4、2 条，各自平均 loss 乘 4/6、2/6。
    不能一律除以 grad_accum_steps，也不能把大小不同的 micro-batch 等权平均。
    由于所有行都贡献 T-1 个预测位置，按序列数加权等价于按有效 token 数加权。
    """
    optimizer.zero_grad(set_to_none=True)
    mean_loss = 0.0
    for start in range(0, len(indices), args.batch_size):
        part = indices[start:start + args.batch_size]
        batch = dataset.batch(part, device)
        with precision_context(device, args.precision):
            loss = model(input_ids=batch, labels=batch).loss
        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError("训练 loss 非有限数，停止更新；请从最近保存的 checkpoint 排查")
        weight = len(part) / len(indices)
        (loss * weight).backward()
        mean_loss += loss.detach().item() * weight
    # 必须在所有 micro-batch 梯度累积完之后裁剪，然后仅更新一次参数。
    # error_if_nonfinite=True 防止 NaN/Inf 梯度进入 AdamW 状态。
    norm = clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return mean_loss, float(norm)


@torch.inference_mode()
def evaluate(model, dataset, args, device):
    """固定顺序验证，按预测位置数加权，包含末尾不足 batch-size 的序列。

    eval() 控制 dropout 等模块行为，inference_mode() 关闭梯度记录；两者用途不同。
    验证不推进训练数据游标。退出后恢复原来的 train/eval 状态。
    """
    was_training = model.training
    model.eval()
    count = len(dataset) if args.eval_batches == 0 else min(len(dataset), args.eval_batches * args.batch_size)
    weighted_loss = 0.0
    try:
        for start in range(0, count, args.batch_size):
            batch = dataset.batch(slice(start, min(start + args.batch_size, count)), device)
            with precision_context(device, args.precision):
                loss = model(input_ids=batch, labels=batch).loss
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError("验证 loss 非有限数")
            weighted_loss += loss.item() * len(batch)
    finally:
        model.train(was_training)
    return weighted_loss / count, count


def epoch_order(size: int, seed: int, epoch: int) -> np.ndarray:
    # 独立 CPU Generator 不消耗模型的随机状态。epoch 改变时顺序改变；恢复时用
    # 相同 seed+epoch 重建排列，再从 next_sequence 接着取，不从 epoch 开头重复读。
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator).numpy()


def rng_state(device):
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(path, model, optimizer, scheduler, progress, contract, tokenizer_path, device):
    """只在完成整个 optimizer update 后保存，不保存半个累积窗口的梯度。

    先写临时文件，再替换 checkpoint.pt；写入失败时保留上一个完整 checkpoint。
    此处保存 state_dict 而非整个模型对象，重新加载时由配置重建模型。
    数据指纹、训练约定和数据游标也一起保存，避免“能加载权重，却悄悄换了实验”。
    """
    checkpoint = {
        "format_version": 1, "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "progress": dict(progress),
        "contract": contract, "rng_state": rng_state(device),
        "tokenizer_path": str(tokenizer_path.resolve()), "torch_version": str(torch.__version__),
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def validate_progress(progress, rows: int, effective_batch: int, epochs: int):
    epoch, cursor = progress["epoch"], progress["next_sequence"]
    if not 0 <= epoch <= epochs or not 0 <= cursor < rows:
        raise ValueError("checkpoint 的数据游标越界")
    if cursor % effective_batch or (epoch == epochs and cursor != 0):
        raise ValueError("checkpoint 不在完整 optimizer update 边界")
    expected_step = epoch * math.ceil(rows / effective_batch) + cursor // effective_batch
    if progress["step"] != expected_step:
        raise ValueError("checkpoint step 与数据游标不一致")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"CUDA device={device} name={torch.cuda.get_device_name(device)} "
              f"visible_count={torch.cuda.device_count()} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 模式需要支持 BF16 的 CUDA GPU；CPU 调试请显式使用 --precision fp32")

    # YAML 仅保存完整的架构字段，不做递归配置继承；训练超参数直接来自命令行。
    raw_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    config = MiniLlamaConfig.from_dict(raw_config["model_config"])
    datasets, data_fingerprint, tokenizer_path = load_data(args.data_dir, config)
    sequence_length = data_fingerprint["sequence_length"]
    rows = len(datasets["train"])
    effective_batch = args.batch_size * args.grad_accum_steps
    total_steps = math.ceil(rows / effective_batch) * args.epochs
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup-steps 必须小于完整训练步数；小数据调试可设为 0")
    # stop_after_steps 只控制本次进程的停止点，不出现在训练约定里。恢复时可以
    # 延后/去掉停止点，但不允许改 batch、epochs 或 LR，否则就不是接着同一实验训练。
    recipe_names = ("batch_size", "grad_accum_steps", "epochs", "learning_rate", "min_lr_ratio",
                    "warmup_steps", "weight_decay", "beta1", "beta2", "max_grad_norm",
                    "seed", "precision", "eval_batches")
    contract = {"model_config": config.to_dict(), "data": data_fingerprint,
                "recipe": {name: getattr(args, name) for name in recipe_names},
                "device_type": device.type, "total_steps": total_steps}
    checkpoint = None
    if args.resume is not None:
        # checkpoint 只含 Tensor、字典及基本类型，显式启用受限的 weights_only 加载。
        # 先在 CPU 加载，防止一次性把整个优化器 checkpoint 都灌进 GPU 造成显存峰值。
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint["contract"] != contract:
            raise ValueError("恢复失败：模型、数据内容或训练约定与 checkpoint 不一致")

    torch.manual_seed(args.seed)
    model = MiniLlamaForCausalLM(config).to(device)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda index: lr_factor(index, total_steps, args.warmup_steps, args.min_lr_ratio))
    progress = {"step": 0, "epoch": 0, "next_sequence": 0, "tokens_seen": 0,
                "last_train_loss": None, "initial_validation_loss": None,
                "validation_loss": None, "validation_step": None}
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # 先构造 scheduler，再加载优化器和 scheduler 状态，避免 scheduler 初始化
        # 改写 checkpoint 已保存的 LR。恢复后 optimizer 中保存的是下一步将使用的 LR。
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        progress = checkpoint["progress"]
        validate_progress(progress, rows, effective_batch, args.epochs)
        if scheduler.last_epoch != progress["step"]:
            raise ValueError("学习率进度与 optimizer step 不一致")
        if progress["tokens_seen"] != (progress["epoch"] * rows + progress["next_sequence"]) * sequence_length:
            raise ValueError("checkpoint 的 tokens_seen 与数据游标不一致")

    stop_step = min(total_steps, args.stop_after_steps or total_steps)
    if stop_step <= progress["step"]:
        raise ValueError("停止步数必须大于 checkpoint 已完成的步数")
    output = (args.output_dir or (args.resume.parent if args.resume else PROJECT_ROOT / "runs/m01_39m")).resolve()
    if args.resume is None or output != args.resume.resolve().parent:
        output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "checkpoint.pt"
    # 恢复时清除旧日志中晚于 checkpoint 的 step，避免崩溃前已记录但未保存的曲线
    # 与重新执行的曲线混在一起。指定另一个新 output-dir 也可从 checkpoint 分叉调试。
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=progress["step"] + 1 if checkpoint else None)
    if checkpoint is not None:
        # 模型重建会消耗随机数，所以必须在初始化、加载完成后再恢复 CPU/CUDA RNG。
        restore_rng(checkpoint["rng_state"], device)
        del checkpoint

    started = time.perf_counter()
    start_step = progress["step"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"parameters={model.num_parameters():,} device={device} precision={args.precision} "
          f"total_steps={total_steps} start_step={start_step} stop_step={stop_step} "
          f"effective_batch={effective_batch} sequences", flush=True)

    def validate_and_log():
        loss, count = evaluate(model, datasets["validation"], args, device)
        progress["validation_loss"] = loss
        progress["validation_step"] = progress["step"]
        if progress["initial_validation_loss"] is None:
            progress["initial_validation_loss"] = loss
        writer.add_scalar("validation/loss", loss, progress["step"])
        writer.add_scalar("validation/prediction_tokens", count * (sequence_length - 1), progress["step"])
        print(f"validation step={progress['step']} loss={loss:.6f} sequences={count}", flush=True)

    try:
        if progress["step"] == 0:
            validate_and_log()
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, progress, contract, tokenizer_path, device)
        model.train()
        order_epoch = None
        order = None
        while progress["step"] < stop_step:
            if order_epoch != progress["epoch"]:
                order_epoch = progress["epoch"]
                order = epoch_order(rows, args.seed, order_epoch)
            cursor = progress["next_sequence"]
            end = min(cursor + effective_batch, rows)
            indices = order[cursor:end]
            # CUDA 是异步执行，计时前后同步，tokens/s 才包含实际 GPU 计算时间。
            # 该指标只统计训练更新，不包含独立验证和 checkpoint 写盘耗时。
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            loss, norm = train_update(model, optimizer, datasets["train"], indices, args, device)
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started

            # 只在参数更新完成后推进游标。epoch 结束即转到下一轮的起点，checkpoint
            # 记录的永远是“下一次该从哪里读”，而不是刚用过的最后一个 micro-batch。
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
                       "tokens_per_second": len(indices) * sequence_length / update_seconds}
            if device.type == "cuda":
                metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, step)
            if step == 1 or step % args.log_every == 0 or step == stop_step:
                print(f"step={step}/{total_steps} loss={loss:.6f} lr={learning_rate:.3e} "
                      f"grad_norm={norm:.4f} tokens/s={metrics['tokens_per_second']:.0f}", flush=True)
            if step % args.eval_every == 0 or step == stop_step:
                validate_and_log()
            if step % args.save_every == 0 or step == stop_step:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, progress, contract, tokenizer_path, device)
                writer.flush()
    finally:
        # 异常或 Ctrl+C 不保存半更新状态；保留最近一次完整 checkpoint 供 --resume。
        writer.close()

    summary = {"status": "complete" if progress["step"] == total_steps else "paused",
               "model_parameters": model.num_parameters(), "progress": progress,
               "contract": contract, "device": str(device), "start_step": start_step,
               "session_elapsed_seconds": round(time.perf_counter() - started, 2),
               "prediction_tokens_seen": progress["tokens_seen"] // sequence_length * (sequence_length - 1),
               "checkpoint": str(checkpoint_path), "tensorboard": str(output / "tensorboard")}
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    print(f"{summary['status']}: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
