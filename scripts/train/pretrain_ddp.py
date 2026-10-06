#!/usr/bin/env python3
"""M3 单机单卡/多卡预训练：理解 DDP 的数据划分、梯度平均与恢复。

双卡启动：torchrun --standalone --nproc_per_node=2 scripts/train/pretrain_ddp.py ...
单卡启动：python scripts/train/pretrain_ddp.py --grad-accum-steps 4 ...

默认使用 M2 的 99M、LR=1e-3、warmup=300；默认每卡 batch=4、累积=2，
双卡的全局 batch 为 16。单卡对照必须把累积改成 4。step 始终指一次参数更新，
不是一个 micro-batch，也不是所有 rank 的更新次数相加。

复用 pretrain.py 的只读数据、配置校验、优化器和 LR 函数，保留原单卡脚本。
本入口仅支持同一 world size/训练约定恢复；不支持跨卡数恢复、FSDP 或 FP16。
CUDA 使用 NCCL；CPU/FP32 使用 Gloo，供小模型正确性测试。

DDP 是数据并行：每个进程都有完整模型、梯度和优化器，不是把模型拆成几份。
DDP 负责同步梯度，不负责分配样本；因此下面仍显式实现数据切分和统计汇总。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train import pretrain as single  # noqa: E402
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402


def build_parser(description=None, *, device_help="cpu 或 cuda；torchrun 下由 LOCAL_RANK 绑定 GPU"):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/tokenized/m01_fineweb_100m")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/model/p099m.yaml")
    parser.add_argument("--output-dir", type=Path, help="新训练必须指定，已有目录拒绝覆盖")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda", help=device_help)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4, help="每个 rank 的 micro-batch 序列数")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="每个 rank 的累积次数；双卡 2，单卡对照 4")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-sequences", type=int,
                        help="每轮仅训练 train.bin 的前 N 条序列，并在这 N 条内打乱；默认使用全部")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adamw", "muon", "muonh", "adamh"), default="adamw")
    parser.add_argument("--muon-lr", type=float,
                        help="Muon 隐藏层矩阵的峰值学习率，默认 0.02")
    parser.add_argument("--adamw-lr", type=float,
                        help="混合优化器中 embedding/head/norm 的 AdamW 峰值学习率；默认沿用 --learning-rate")
    parser.add_argument("--hyperball-lr", type=float,
                        help="MuonH/AdamH 隐藏层矩阵的峰值相对更新长度；必须显式指定")
    parser.add_argument("--muon-momentum", type=float, help="Muon 动量，默认 0.95")
    parser.add_argument("--muon-ns-steps", type=int, help="Newton-Schulz 迭代次数，默认 5")
    parser.add_argument("--record-update-norms", action="store_true",
                        help="按 --log-every 记录参数、实际更新范数及其比值；Muon 入口默认启用")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="0 表示全量验证；正数限制全局前 N×batch-size 条，不随卡数翻倍")
    parser.add_argument("--stop-after-steps", type=int, help="暂停的总更新步数，不改变完整 LR 计划")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4, help="每个 rank 的 CPU 线程数")
    return parser


def parse_args(parser=None):
    parser = parser or build_parser()
    args = parser.parse_args()
    for name in ("batch_size", "grad_accum_steps", "epochs", "eval_every", "save_every", "log_every", "cpu_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正数")
    for name in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} 必须为有限正数")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay 必须为有限非负数")
    if not 0 <= args.min_lr_ratio <= 1 or not all(0 <= b < 1 for b in (args.beta1, args.beta2)):
        parser.error("min-lr-ratio 必须在 [0,1]，betas 必须在 [0,1)")
    if min(args.warmup_steps, args.eval_batches, args.seed) < 0:
        parser.error("warmup-steps、eval-batches、seed 不能为负数")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        parser.error("stop-after-steps 必须为正数")
    if args.train_sequences is not None and args.train_sequences <= 0:
        parser.error("train-sequences 必须为正数")
    if args.adamw_lr is not None and (not math.isfinite(args.adamw_lr) or args.adamw_lr <= 0):
        parser.error("adamw-lr 必须为有限正数")
    if args.optimizer in ("muon", "muonh"):
        if args.optimizer == "muon":
            args.muon_lr = 0.02 if args.muon_lr is None else args.muon_lr
        args.muon_momentum = 0.95 if args.muon_momentum is None else args.muon_momentum
        args.muon_ns_steps = 5 if args.muon_ns_steps is None else args.muon_ns_steps
        if args.optimizer == "muon" and (not math.isfinite(args.muon_lr) or args.muon_lr <= 0):
            parser.error("muon-lr 必须为有限正数")
        if not math.isfinite(args.muon_momentum) or not 0 <= args.muon_momentum < 1:
            parser.error("muon-momentum 必须在 [0,1) 内")
        if args.muon_ns_steps <= 0:
            parser.error("muon-ns-steps 必须为正数")
        if args.optimizer == "muonh" and args.muon_lr is not None:
            parser.error("MuonH 使用 --hyperball-lr，不使用 --muon-lr")
    elif any(value is not None for value in (args.muon_lr, args.muon_momentum, args.muon_ns_steps)):
        parser.error("Muon 专用参数只在 --optimizer muon/muonh 时生效")
    if args.optimizer in ("muonh", "adamh"):
        if args.hyperball_lr is None or not math.isfinite(args.hyperball_lr) or args.hyperball_lr <= 0:
            parser.error("MuonH/AdamH 必须指定有限正数 --hyperball-lr")
    elif args.hyperball_lr is not None:
        parser.error("--hyperball-lr 只在 MuonH/AdamH 模式生效")
    if args.optimizer == "adamw" and args.adamw_lr is not None:
        parser.error("AdamW 模式使用 --learning-rate，不使用 --adamw-lr")
    if args.output_dir is None and args.resume is None:
        parser.error("新训练必须指定 --output-dir")
    return args


@dataclass(frozen=True)
class Context:
    """rank 是全局进程编号，world_size 是总进程数；本脚本每进程使用一张 GPU。"""

    rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self):
        return self.world_size > 1


def setup_distributed(requested: str) -> Context:
    """torchrun 启动多个独立 Python 进程，并给每个进程注入 rank 环境变量。

    LOCAL_RANK 是本机可见设备的逻辑编号。例如 CUDA_VISIBLE_DEVICES=1,0 时，
    local rank 0 使用物理卡 1，local rank 1 使用物理卡 0。
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("无效的 RANK / WORLD_SIZE / LOCAL_RANK")
    requested_device = torch.device(requested)
    if requested_device.type == "cuda" and world_size > 1:
        if requested_device.index is not None:
            raise ValueError("多卡请使用 --device cuda，由 LOCAL_RANK 选卡，不要显式传 cuda:N")
        requested = f"cuda:{local_rank}"
    device = single.resolve_device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"rank={rank}/{world_size} local_rank={local_rank} device={device} "
              f"name={torch.cuda.get_device_name(device)} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                init_method="env://", timeout=timedelta(minutes=5))
    return Context(rank, world_size, device)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def rank_zero_call(ctx, action):
    """只让 rank 0 写文件，并把失败传给所有 rank，避免其他进程一直等保存完成。"""
    if not ctx.distributed:
        action()
        return
    error = [None]
    if ctx.rank == 0:
        try:
            action()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"rank 0 操作失败：{error[0]}")


def _orthogonalize_muon(update: torch.Tensor, steps: int) -> torch.Tensor:
    """原始 Muon 的五次多项式 Newton-Schulz 近似；CPU 测试使用 FP32。"""
    # 公式与矩阵形状缩放参照 https://github.com/KellerJordan/Muon (MIT)。
    x = update.to(torch.bfloat16 if update.is_cuda and torch.cuda.is_bf16_supported()
                  else torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    if transposed:
        x = x.T
    x = x * max(1.0, update.shape[0] / update.shape[1]) ** 0.5
    return x.to(update.dtype)


def split_hidden_and_aux_parameters(model):
    """Transformer 二维矩阵与其余参数互斥分组，排除共享 embedding/head。"""
    hidden, aux_decay, aux_no_decay = [], [], []
    embedding_id = id(model.embed_tokens.weight)
    head_id = id(model.lm_head.weight)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("layers.") and parameter.ndim == 2 and id(parameter) not in (embedding_id, head_id):
            hidden.append(parameter)
        elif parameter.ndim >= 2:
            aux_decay.append(parameter)
        else:
            aux_no_decay.append(parameter)
    if not hidden:
        raise ValueError("Muon/Hyperball 需要至少一个 Transformer 隐藏层二维参数")
    registered = [*hidden, *aux_decay, *aux_no_decay]
    expected = [p for p in model.parameters() if p.requires_grad]
    if len(registered) != len(expected) or {id(p) for p in registered} != {id(p) for p in expected}:
        raise RuntimeError("优化器参数分组存在重复或遗漏")
    return hidden, aux_decay, aux_no_decay


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """隐藏层二维权重用 Muon，其余参数用 AdamW；共享参数只注册一次。"""

    def __init__(self, model, args):
        hidden, aux_decay, aux_no_decay = split_hidden_and_aux_parameters(model)
        adamw_lr = args.adamw_lr if args.adamw_lr is not None else args.learning_rate
        groups = [
            {"params": hidden, "use_muon": True, "lr": args.muon_lr,
             "momentum": args.muon_momentum, "ns_steps": args.muon_ns_steps,
             "weight_decay": args.weight_decay},
            {"params": aux_decay, "use_muon": False, "lr": adamw_lr,
             "betas": (args.beta1, args.beta2), "eps": 1e-8, "weight_decay": args.weight_decay},
            {"params": aux_no_decay, "use_muon": False, "lr": adamw_lr,
             "betas": (args.beta1, args.beta2), "eps": 1e-8, "weight_decay": 0.0},
        ]
        super().__init__(groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                if group["use_muon"]:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    momentum = state["momentum_buffer"]
                    momentum.lerp_(grad, 1 - group["momentum"])
                    direction = grad.lerp(momentum, group["momentum"])
                    update = _orthogonalize_muon(direction, group["ns_steps"])
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denominator = state["exp_avg_sq"].sqrt() / math.sqrt(1 - beta2 ** state["step"])
                    denominator.add_(group["eps"])
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.addcdiv_(state["exp_avg"], denominator,
                                        value=-group["lr"] / (1 - beta1 ** state["step"]))
        return loss


@torch.no_grad()
def _record_update_norms(before, output):
    """记录参数更新后的 L2 范数，以及包含 weight decay 的实际参数位移。"""
    sums = {}
    for kind, parameter, previous in before:
        if kind not in sums:
            zero = torch.zeros((), dtype=torch.float64, device=parameter.device)
            sums[kind] = [zero.clone(), zero.clone()]
        sums[kind][0] += parameter.detach().square().sum(dtype=torch.float64)
        sums[kind][1] += (parameter.detach() - previous).square().sum(dtype=torch.float64)
    all_parameters = sum((pair[0] for pair in sums.values()))
    all_updates = sum((pair[1] for pair in sums.values()))
    for name, parameter_sq, update_sq in [
        ("", all_parameters, all_updates),
        *((f"{kind}_", values[0], values[1]) for kind, values in sums.items()),
    ]:
        parameter_norm = math.sqrt(parameter_sq.item())
        update_norm = math.sqrt(update_sq.item())
        output[f"{name}parameter_norm"] = parameter_norm
        output[f"{name}update_norm"] = update_norm
        output[f"{name}update_to_parameter_ratio"] = update_norm / parameter_norm if parameter_norm else 0.0


def train_update(model, optimizer, dataset, global_indices, args, ctx, norm_metrics=None):
    """同一全局 batch 先分给 rank，再在每个 rank 内做 micro-batch 累积。

    标准 DDP 默认对各 rank 梯度取平均。若全局有 N 条样本，本地某个
    micro-batch 有 m 条，它的平均 loss 必须乘 world_size*m/N：
    DDP 的 1/world_size 与这个倍率抵消后，每条真实样本的权重恰好为 1/N。
    这也适用于 rank 间样本数不等的尾部，不能简单平均各 rank 的平均 loss。
    """
    count = len(global_indices)
    if count == 0:
        raise ValueError("全局 batch 不能为空")
    local_indices = np.array_split(global_indices, ctx.world_size)[ctx.rank]
    # 所有 rank 执行相同数量的 forward/backward，且最后一次同时触发同步。
    # 极小尾部可能使某 rank 没有真实样本：用一个零权重样本参与计算图，
    # 不计入 loss/tokens_seen。当前模型无 BatchNorm；这是本模型专用的尾部策略。
    rounds = math.ceil(math.ceil(count / ctx.world_size) / args.batch_size)
    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = 0.0
    for index in range(rounds):
        part = local_indices[index * args.batch_size:(index + 1) * args.batch_size]
        batch = dataset.batch(part if len(part) else global_indices[:1], ctx.device)
        sync_context = model.no_sync() if ctx.distributed and index < rounds - 1 else nullcontext()
        # no_sync 必须同时包住 forward 和 backward，只包 backward 不足以关闭同步。
        with sync_context:
            with single.precision_context(ctx.device, args.precision):
                loss = model(input_ids=batch, labels=batch).loss
            if loss is None:
                raise RuntimeError("模型没有返回 causal LM loss")
            (loss * (ctx.world_size * len(part) / count)).backward()
        local_loss_sum += loss.detach().item() * len(part)

    total = torch.tensor(local_loss_sum, dtype=torch.float64, device=ctx.device)
    if ctx.distributed:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    # 所有 rank 得到同一个全局 loss，统一报错。非有限梯度在下面裁剪时也会报错。
    if not torch.isfinite(total).item():
        raise RuntimeError("训练 loss 出现 NaN/Inf，停止更新")
    # 此时最后一次 backward 已同步整个累积梯度；先同步再裁剪，不能各卡先裁剪。
    norm = clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
    before = None
    if norm_metrics is not None:
        before = [(group.get("norm_kind", "muon" if group.get("use_muon") else "adamw"), parameter,
                   parameter.detach().clone())
                  for group in optimizer.param_groups for parameter in group["params"]]
    optimizer.step()
    if before is not None:
        _record_update_norms(before, norm_metrics)
    return total.item() / count, float(norm)


@torch.inference_mode()
def evaluate(model, dataset, args, ctx):
    """无重复地划分验证数据，汇总 loss 总和与样本数，而非平均 rank 的均值。

    每条序列预测位置数均为 T-1，所以按序列加权等价于按预测 token 加权。
    使用原始模型 forward，避免某些 rank 多跑一个尾部 batch 时触发 DDP 通信。
    """
    raw_model = unwrap(model)
    was_training = model.training
    model.eval()
    count = len(dataset) if args.eval_batches == 0 else min(len(dataset), args.eval_batches * args.batch_size)
    start = count * ctx.rank // ctx.world_size
    end = count * (ctx.rank + 1) // ctx.world_size
    loss_sum = 0.0
    try:
        for cursor in range(start, end, args.batch_size):
            batch = dataset.batch(slice(cursor, min(cursor + args.batch_size, end)), ctx.device)
            with single.precision_context(ctx.device, args.precision):
                loss = raw_model(input_ids=batch, labels=batch).loss
            loss_sum += loss.item() * len(batch)
        sums = torch.tensor([loss_sum, end - start], dtype=torch.float64, device=ctx.device)
        if ctx.distributed:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        if not torch.isfinite(sums).all().item() or sums[1].item() != count or count == 0:
            raise RuntimeError("验证 loss 非有限或样本计数错误")
        return sums[0].item() / count, count
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, scheduler, progress, contract, tokenizer_path, ctx):
    """所有 rank 都要调用：先收集各自 RNG，再只让 rank 0 原子写入文件。

    DDP 同步后各 rank 的模型/优化器相同，因此保存一份即可；RNG 可能不同，
    必须每个 rank 各存一份。权重用原始模型的 key，不带 'module.' 前缀，
    通用 pretrained_checkpoint_demo.py 可直接读取。
    """
    local_rng = single.rng_state(ctx.device)
    states = [None] * ctx.world_size
    if ctx.distributed:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng

    def write():
        checkpoint = {
            "format_version": 2, "model_config": unwrap(model).config.to_dict(),
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "progress": dict(progress), "contract": contract, "rng_states": states,
            "tokenizer_path": str(tokenizer_path.resolve()), "torch_version": str(torch.__version__),
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    rank_zero_call(ctx, write)


def load_resume(path, contract, ctx):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2 or checkpoint.get("contract") != contract:
        raise ValueError("恢复失败：需要本 DDP 脚本的 v2 checkpoint，且卡数、模型、数据和训练约定必须一致")
    if len(checkpoint.get("rng_states", [])) != ctx.world_size:
        raise ValueError("checkpoint 缺少各 rank 的 RNG 状态")
    return checkpoint


def run(args, ctx):
    if args.precision == "bf16" and (ctx.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 需要支持它的 CUDA GPU；CPU 测试使用 --precision fp32")
    config = MiniLlamaConfig.from_dict(yaml.safe_load(args.model_config.read_text())["model_config"])
    datasets, fingerprint, tokenizer_path = single.load_data(args.data_dir, config)
    sequence_length = fingerprint["sequence_length"]
    available_rows = len(datasets["train"])
    if args.train_sequences is not None and args.train_sequences > available_rows:
        raise ValueError(f"train-sequences={args.train_sequences} 超过训练集序列数 {available_rows}")
    rows = available_rows if args.train_sequences is None else args.train_sequences
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
                "device_type": ctx.device.type, "total_steps": total_steps}
    # 默认模式保持原 checkpoint contract 不变；显式限制训练集时把范围纳入恢复约定。
    if args.train_sequences is not None:
        contract["recipe"]["train_sequences"] = args.train_sequences
    if args.optimizer == "muon":
        contract["recipe"].update({"optimizer": "muon", "muon_lr": args.muon_lr,
                                   "adamw_lr": args.adamw_lr if args.adamw_lr is not None else args.learning_rate,
                                   "muon_momentum": args.muon_momentum,
                                   "muon_ns_steps": args.muon_ns_steps})
    elif args.optimizer in ("muonh", "adamh"):
        contract["recipe"].update({"optimizer": args.optimizer,
                                   "hyperball_lr": args.hyperball_lr,
                                   "adamw_lr": args.adamw_lr if args.adamw_lr is not None else args.learning_rate})
        if args.optimizer == "muonh":
            contract["recipe"].update({"muon_momentum": args.muon_momentum,
                                       "muon_ns_steps": args.muon_ns_steps})
    # 也核对 rank 间的数据与命令行，防止同名文件在不同挂载点下内容不同。
    if ctx.distributed:
        contracts = [None] * ctx.world_size
        dist.all_gather_object(contracts, contract)
        if any(item != contract for item in contracts):
            raise ValueError("各 rank 的模型、数据或训练约定不一致")
    checkpoint = load_resume(args.resume, contract, ctx) if args.resume else None

    # 先用相同 seed 初始化。DDP 构造时还会从 rank 0 同步权重，确保一致起点。
    torch.manual_seed(args.seed)
    raw_model = MiniLlamaForCausalLM(config).to(ctx.device)
    if args.optimizer == "muon":
        optimizer = MuonWithAuxAdamW(raw_model, args)
    elif args.optimizer in ("muonh", "adamh"):
        from scripts.train.pretrain_hyperball import HyperballWithAuxAdamW
        optimizer = HyperballWithAuxAdamW(raw_model, args)
    else:
        optimizer = single.make_optimizer(raw_model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda i: single.lr_factor(i, total_steps, args.warmup_steps, args.min_lr_ratio))
    progress = {"step": 0, "epoch": 0, "next_sequence": 0, "tokens_seen": 0,
                "last_train_loss": None, "initial_validation_loss": None,
                "validation_loss": None, "validation_step": None}
    if checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        progress = checkpoint["progress"]
        single.validate_progress(progress, rows, global_batch, args.epochs)
        if scheduler.last_epoch != progress["step"]:
            raise ValueError("学习率进度与 step 不一致")
        if progress["tokens_seen"] != (progress["epoch"] * rows + progress["next_sequence"]) * sequence_length:
            raise ValueError("tokens_seen 与全局数据游标不一致")
    model = raw_model
    if ctx.distributed:
        # RoPE buffer 是固定频率，各 rank 构造相同；无须每次 forward 广播。
        # 当前所有可训练参数都参与 loss，不启用 find_unused_parameters。
        # 各 rank 独立执行相同优化器更新；DDP 不是每步把 rank 0 权重复制过来，
        # 而是在 backward 中同步梯度，使同起点、同优化器的各副本保持相同参数。
        model = DDP(raw_model, device_ids=[ctx.device.index] if ctx.device.type == "cuda" else None,
                    broadcast_buffers=False)
    stop_step = min(total_steps, args.stop_after_steps or total_steps)
    if stop_step <= progress["step"]:
        raise ValueError("停止步数必须大于 checkpoint 已完成的步数")
    output = (args.output_dir or args.resume.parent).resolve()
    if args.resume is None or output != args.resume.resolve().parent:
        rank_zero_call(ctx, lambda: output.mkdir(parents=True, exist_ok=False))
    checkpoint_path = output / "checkpoint.pt"
    writer = None
    try:
        # 每个 rank 写同一文件会造成竞态，所以仅 rank 0 创建 SummaryWriter。
        if ctx.rank == 0:
            writer = SummaryWriter(str(output / "tensorboard"),
                                   purge_step=progress["step"] + 1 if checkpoint else None)
        if checkpoint:
            # 模型重建会消耗 RNG。全部构造完再恢复，不改变后续 dropout 等随机序列。
            single.restore_rng(checkpoint["rng_states"][ctx.rank], ctx.device)
            del checkpoint
        started = time.perf_counter()
        start_step = progress["step"]
        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)
        if ctx.rank == 0:
            print(f"parameters={raw_model.num_parameters():,} world_size={ctx.world_size} "
                  f"global_batch={global_batch} total_steps={total_steps} start_step={start_step} "
                  f"stop_step={stop_step} precision={args.precision} optimizer={args.optimizer}", flush=True)

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
                # 各 rank 用同一全局排列、同一游标，再在 train_update 内切分。
                # 仅相同 seed + 各自独立取 batch 不够，会把数据重复训练 world_size 次。
                order = single.epoch_order(rows, args.seed, order_epoch)
            cursor = progress["next_sequence"]
            end = min(cursor + global_batch, rows)
            indices = order[cursor:end]
            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)
            update_started = time.perf_counter()
            if ctx.distributed:
                dist.barrier()
            learning_rate = optimizer.param_groups[0]["lr"]
            adamw_learning_rate = (optimizer.param_groups[1]["lr"]
                                   if args.optimizer in ("muon", "muonh", "adamh") else None)
            next_step = progress["step"] + 1
            norm_metrics = ({} if args.record_update_norms and
                            (next_step == 1 or next_step % args.log_every == 0 or next_step == stop_step)
                            else None)
            loss, norm = train_update(model, optimizer, datasets["train"], indices, args, ctx, norm_metrics)
            scheduler.step()
            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)
            elapsed = torch.tensor(time.perf_counter() - update_started, dtype=torch.float64, device=ctx.device)
            if ctx.distributed:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            # 分母取最慢 rank，分子已经是全局 tokens，不能再乘一次 world_size。
            # 含更新前 barrier、forward/backward、梯度同步和优化器；不含验证/写盘。
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
            if args.optimizer in ("muon", "muonh", "adamh"):
                metrics[f"{args.optimizer}_learning_rate"] = learning_rate
                metrics["adamw_learning_rate"] = adamw_learning_rate
            if norm_metrics is not None:
                metrics.update(norm_metrics)
            if ctx.device.type == "cuda":
                memory = torch.tensor(torch.cuda.max_memory_allocated(ctx.device) / 1024**3, device=ctx.device)
                memories = [torch.zeros_like(memory) for _ in range(ctx.world_size)]
                if ctx.distributed:
                    dist.all_gather(memories, memory)
                else:
                    memories[0] = memory
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

        session_seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=ctx.device)
        if ctx.distributed:
            dist.all_reduce(session_seconds, op=dist.ReduceOp.MAX)
        summary = {"status": "complete" if progress["step"] == total_steps else "paused",
                   "model_parameters": raw_model.num_parameters(), "progress": progress, "contract": contract,
                   "device": str(ctx.device), "world_size": ctx.world_size, "start_step": start_step,
                   "session_elapsed_seconds": round(session_seconds.item(), 2),
                   "prediction_tokens_seen": progress["tokens_seen"] // sequence_length * (sequence_length - 1),
                   "checkpoint": str(checkpoint_path), "tensorboard": str(output / "tensorboard")}

        def write_summary():
            temporary = output / "summary.json.tmp"
            temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output / "summary.json")
            print(f"{summary['status']}: {checkpoint_path}", flush=True)
        rank_zero_call(ctx, write_summary)
    finally:
        if writer:
            writer.close()


def main(args=None):
    args = parse_args() if args is None else args
    torch.set_num_threads(args.cpu_threads)
    ctx = setup_distributed(args.device)
    try:
        run(args, ctx)
    finally:
        # 异常时不增加 barrier，以免已经退出一个 rank 后其他 rank 永远等待。
        # torchrun 负责在一个 worker 失败时终止其他 worker；恢复使用最近完整更新。
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
