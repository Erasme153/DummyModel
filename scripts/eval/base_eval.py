#!/usr/bin/env python3
"""Evaluate a pretraining checkpoint on packed validation tokens or zero-shot tasks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402
from scripts.train.pretrain import PackedDataset, file_sha256, precision_context  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("loss", "benchmark"):
        part = sub.add_parser(name)
        part.add_argument("--checkpoint", type=Path, required=True)
        part.add_argument("--device", default="cuda:0")
        part.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        part.add_argument("--batch-size", type=int, default=4)
        if name == "loss":
            part.add_argument("--data-dir", type=Path, required=True)
            part.add_argument("--max-sequences", type=int, default=0,
                              help="0 使用完整验证集；正数仅用于小样本检查")
            part.add_argument("--output", type=Path, help="可选：保存 JSON 结果")
        else:
            part.add_argument("--tasks", default="hellaswag,arc_easy")
            part.set_defaults(batch_size=1)
            part.add_argument("--limit", type=int, default=0,
                              help="0 使用完整 benchmark；正数仅用于小样本检查")
            part.add_argument("--output", type=Path, required=True)
    return result


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # 本项目自产的 checkpoint 包含非 Tensor 的配置和训练状态。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") == 4 and checkpoint.get("contract", {}).get("parallelism") == "ep2":
        # EP metadata points at two local shards. Shared keys must agree; expert
        # keys retain their global IDs, so the union is a regular model state.
        merged = {}
        for name in checkpoint["shards"]:
            shard = torch.load(checkpoint_path.parent / name, map_location="cpu",
                               weights_only=True, mmap=True)["model_state_dict"]
            for key, value in shard.items():
                if key in merged:
                    if not torch.equal(merged[key], value):
                        raise ValueError(f"EP 共享权重不一致：{key}")
                else:
                    merged[key] = value
        checkpoint["model_state_dict"] = merged
    config = MiniLlamaConfig.from_dict(checkpoint["model_config"])
    model = MiniLlamaForCausalLM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, config, checkpoint


def evaluate_loss(args, model, config, checkpoint, device):
    metadata = json.loads((args.data_dir / "summary.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or metadata["format"]["dtype"] != "<u2":
        raise ValueError("验证数据必须完整，且为小端 uint16")
    length = metadata["format"]["sequence_length"]
    if length > config.max_position_embeddings:
        raise ValueError("验证序列超过模型最大上下文")
    tokenizer_path = args.data_dir / metadata["tokenizer"]["file"]
    tokenizer_sha = file_sha256(tokenizer_path)
    if tokenizer_sha != metadata["tokenizer"]["sha256"]:
        raise ValueError("验证集 tokenizer 指纹不符")
    trained_tokenizer_sha = checkpoint["contract"]["data"]["tokenizer_sha256"]
    if tokenizer_sha != trained_tokenizer_sha or config.vocab_size != metadata["tokenizer"]["vocab_size"]:
        raise ValueError("checkpoint 与验证集使用了不同 tokenizer")
    path = args.data_dir / "validation.bin"
    dataset = PackedDataset(path, metadata["splits"]["validation"], length, config.vocab_size)
    validation_sha = file_sha256(path)
    count = len(dataset) if args.max_sequences == 0 else min(args.max_sequences, len(dataset))
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, count, args.batch_size):
            batch = dataset.batch(np.arange(start, min(start + args.batch_size, count)), device)
            with precision_context(device, args.precision):
                loss = model(input_ids=batch, labels=batch).loss
            total_loss += float(loss) * len(batch)
    mean_loss = total_loss / count
    return {"checkpoint": str(args.checkpoint.resolve()),
            "validation": str(path.resolve()), "validation_sha256": validation_sha,
            "sequences": count,
            "prediction_tokens": count * (length - 1),
            "loss": mean_loss, "perplexity": math.exp(mean_loss),
            "precision": args.precision}


def run_benchmark(args, model, config, checkpoint, device):
    try:
        import lm_eval
        from lm_eval.api.model import LM
        from lm_eval.utils import handle_non_serializable
    except ImportError as error:
        raise RuntimeError("请先安装评测依赖：python -m pip install lm-eval") from error

    tokenizer_path = Path(checkpoint["tokenizer_path"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise ValueError("checkpoint tokenizer 与模型词表不符")
    if file_sha256(tokenizer_path) != checkpoint["contract"]["data"]["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer 指纹不符")

    class DummyMLM(LM):
        def __init__(self):
            super().__init__()
            self._device = device

        @property
        def batch_size(self):
            return args.batch_size

        @property
        def max_length(self):
            return config.max_position_embeddings

        @property
        def eot_token_id(self):
            return config.eos_token_id

        def loglikelihood(self, requests):
            answers = []
            with torch.inference_mode():
                for request in requests:
                    context, continuation = request.args
                    if not continuation:
                        raise ValueError("benchmark continuation 不能为空")
                    if context:
                        # 与 harness 的 causal _encode_pair 一致：尾部空白划给 continuation。
                        spaces = len(context) - len(context.rstrip())
                        if spaces:
                            continuation = context[-spaces:] + continuation
                            context = context[:-spaces]
                        context_ids = tokenizer.encode(context, add_special_tokens=False).ids
                        joint = tokenizer.encode(context + continuation, add_special_tokens=False).ids
                        target_ids = joint[len(context_ids):]
                    else:
                        context_ids = [config.eos_token_id]
                        target_ids = tokenizer.encode(continuation, add_special_tokens=False).ids
                    if not context_ids:
                        context_ids = [config.eos_token_id]
                    if not target_ids or len(target_ids) >= self.max_length:
                        raise ValueError("benchmark continuation 为空或超过模型上下文")
                    context_ids = context_ids[-(self.max_length - len(target_ids)):]
                    ids = context_ids + target_ids
                    inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                    with precision_context(device, args.precision):
                        logits = model(input_ids=inputs).logits[0]
                    scores = logits[len(context_ids) - 1:].float().log_softmax(dim=-1)
                    targets = torch.tensor(target_ids, dtype=torch.long, device=device)
                    logprob = scores.gather(1, targets[:, None]).sum().item()
                    greedy = bool(torch.all(scores.argmax(dim=-1) == targets).item())
                    answers.append((logprob, greedy))
            return answers

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("此入口只支持所选的选择题任务")

        def generate_until(self, requests):
            raise NotImplementedError("此入口只支持所选的选择题任务")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks or any(task not in {"hellaswag", "arc_easy"} for task in tasks):
        raise ValueError("当前适配器只支持 hellaswag 和 arc_easy")
    result = lm_eval.simple_evaluate(model=DummyMLM(), tasks=tasks, num_fewshot=0,
                                     limit=args.limit or None, batch_size=args.batch_size,
                                     device=str(device), log_samples=False, bootstrap_iters=1000,
                                     random_seed=2026, numpy_random_seed=2026,
                                     torch_random_seed=2026, fewshot_random_seed=2026)
    output = {"checkpoint": str(args.checkpoint.resolve()), "tasks": tasks,
              "num_fewshot": 0, "limit": args.limit or None,
              "precision": args.precision, "harness_version": getattr(lm_eval, "__version__", None),
              "task_versions": result.get("versions"),
              "results": result["results"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False,
                                      default=handle_non_serializable) + "\n", encoding="utf-8")
    return output


def main():
    args = parser().parse_args()
    if args.batch_size <= 0 or (args.command == "loss" and args.max_sequences < 0) or (
        args.command == "benchmark" and args.limit < 0
    ):
        raise ValueError("batch-size 必须为正数；max-sequences/limit 不能为负数")
    if args.command == "benchmark" and args.batch_size != 1:
        raise ValueError("当前 benchmark 适配器逐请求评分，请使用 --batch-size 1")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"评测结果已存在：{args.output}")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device 仅支持 cpu/cuda")
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 需要支持该精度的 CUDA 设备")
    model, config, checkpoint = load_model(args.checkpoint, device)
    result = (evaluate_loss(args, model, config, checkpoint, device)
              if args.command == "loss" else run_benchmark(args, model, config, checkpoint, device))
    if args.command == "loss" and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
