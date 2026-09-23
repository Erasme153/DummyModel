#!/usr/bin/env python3
"""加载 DummyM 的完整 ``.pt`` checkpoint，并对给定 prompt 续写。

这个入口只依赖推理真正需要的三个公共字段：``model_config``、
``model_state_dict`` 和 ``tokenizer_path``。因此它既能读取早期的 M0
tiny-overfit checkpoint，也能读取 M1 及后续由单文件训练入口保存的完整
checkpoint；它不会依赖某个实验特有的语料路径、loss 阈值或 optimizer 状态。

当前生成实现没有 KV cache，每生成一个 token 都会重新计算完整前缀。这个脚本
适合检查 checkpoint 能否正确加载以及输出是否合理，不用于测量生产推理吞吐。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dummym.inference import generate_token_ids  # noqa: E402
from dummym.models.llama_like import (  # noqa: E402
    MiniLlamaConfig,
    MiniLlamaForCausalLM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a DummyM checkpoint and generate a prompt continuation.",
    )
    # checkpoint 必须显式指定。不同实验的路径相似，使用隐式默认值容易把 M1
    # checkpoint 传给 M0 专用验收脚本，或者误以为测试了最新模型。
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="完整单文件 checkpoint.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="覆盖 checkpoint 中记录的 tokenizer 路径，迁移机器时使用",
    )
    parser.add_argument(
        "--prompt",
        default="The small language model",
        help="需要模型续写的文本",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 表示贪心生成，大于 0 表示采样",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="采样只保留概率最高的 K 个 token；0 表示禁用",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="nucleus sampling 阈值；1.0 表示禁用",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、cuda 或 cuda:N；N 是当前进程可见的逻辑编号",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="auto 在 CUDA 上使用 BF16，在 CPU 上使用 FP32",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens 必须大于或等于 0")
    if args.temperature < 0:
        raise ValueError("--temperature 必须大于或等于 0")
    if args.top_k < 0:
        raise ValueError("--top-k 必须大于或等于 0")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p 必须在 (0, 1] 范围内")


def resolve_device(requested: str) -> torch.device:
    """解析并检查设备，CUDA 编号遵循 ``CUDA_VISIBLE_DEVICES`` 后的逻辑编号。"""

    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device 只支持 auto、cpu、cuda 或 cuda:N")
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU 设备不能带编号")
        return device

    if not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 看不到可用 GPU")
    index = 0 if device.index is None else device.index
    visible_count = torch.cuda.device_count()
    if index < 0 or index >= visible_count:
        raise RuntimeError(
            f"CUDA 逻辑编号 {index} 不存在；当前进程只看到 {visible_count} 张 GPU。"
            "设置 CUDA_VISIBLE_DEVICES 后，设备会从 cuda:0 重新编号。"
        )
    return torch.device("cuda", index)


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("当前脚本只在 CUDA 推理时启用 --precision bf16")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16")
        return torch.bfloat16
    return torch.float32


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")

    # .pt 文件基于 pickle，只能加载自己训练或确认可信来源的文件。这里需要读取
    # 不止权重 Tensor 的配置和训练进度，因此显式使用 weights_only=False。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 顶层必须是字典")

    required = {"model_config", "model_state_dict"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(
            "checkpoint 缺少通用推理字段：" + ", ".join(missing)
        )
    return checkpoint


def resolve_tokenizer_path(
    override: Path | None,
    checkpoint: dict[str, Any],
) -> Path:
    """优先使用命令行路径，否则读取 checkpoint 保存的训练 tokenizer。"""

    if override is not None:
        path = override
    else:
        saved_path = checkpoint.get("tokenizer_path")
        if not saved_path:
            raise ValueError(
                "checkpoint 没有 tokenizer_path，请通过 --tokenizer 显式指定"
            )
        path = Path(saved_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"tokenizer 不存在：{path}；如果 checkpoint 来自另一台机器，"
            "请通过 --tokenizer 指向当前机器上的 tokenizer.json"
        )
    return path


def checkpoint_progress(checkpoint: dict[str, Any]) -> tuple[Any, Any]:
    """兼容 M0 的顶层 step 和 M1 以后保存在 progress 中的训练进度。"""

    progress = checkpoint.get("progress")
    if isinstance(progress, dict):
        return progress.get("step"), progress.get("tokens_seen")
    return checkpoint.get("step"), None


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.precision, device)

    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer_path = resolve_tokenizer_path(args.tokenizer, checkpoint)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    config = MiniLlamaConfig.from_dict(checkpoint["model_config"])
    tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab_size != config.vocab_size:
        raise ValueError(
            f"tokenizer 词表大小为 {tokenizer_vocab_size}，"
            f"但模型配置要求 {config.vocab_size}；不能混用 tokenizer"
        )

    # 先在 CPU 上严格加载参数，可以尽早发现缺失、多余或名称不一致的权重；
    # 然后一次性迁移到目标设备与推理精度，避免在 GPU 上保留额外临时副本。
    model = MiniLlamaForCausalLM(config)
    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # 预训练数据直接编码普通文本并用 EOS 分隔文档，因此这里默认不自动添加 BOS
    # 或 chat template。Base model 接受的是文本续写 prompt，而不是聊天消息。
    prompt_ids = tokenizer.encode(
        args.prompt,
        add_special_tokens=False,
    ).ids
    if not prompt_ids:
        raise ValueError("prompt 编码后没有任何 token")
    if len(prompt_ids) >= config.max_position_embeddings:
        raise ValueError(
            f"prompt 有 {len(prompt_ids)} 个 token，但模型上下文长度只有 "
            f"{config.max_position_embeddings}；必须至少留一个位置用于生成"
        )

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    top_k = None if args.top_k == 0 else args.top_k
    all_token_ids = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        eos_token_id=config.eos_token_id,
        generator=generator,
    )

    new_token_ids = all_token_ids[0, len(prompt_ids):].tolist()
    generated_text = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=False,
    )
    full_text = tokenizer.decode(
        all_token_ids[0].tolist(),
        skip_special_tokens=False,
    )
    step, tokens_seen = checkpoint_progress(checkpoint)

    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"tokenizer: {tokenizer_path.resolve()}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"load_result: {load_result}")
    print(f"model_parameters: {model.num_parameters():,}")
    print(f"checkpoint_step: {step}")
    if tokens_seen is not None:
        print(f"checkpoint_tokens_seen: {tokens_seen}")
    print(f"context_length: {config.max_position_embeddings}")
    print(f"prompt_tokens: {len(prompt_ids)}")
    print(f"generated_tokens: {len(new_token_ids)}")
    print(f"prompt: {args.prompt!r}")
    print(f"generated_text: {generated_text!r}")
    print(f"full_text: {full_text!r}")
    print(f"generated_token_ids: {new_token_ids}")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
# python scripts/inference/pretrained_checkpoint_demo.py \
#   --checkpoint runs/m01_39m/checkpoint.pt \
#   --device cuda:0 \
#   --prompt "The future of artificial intelligence" \
#   --max-new-tokens 32 \
#   --temperature 0.9 \
#   --top-k 0 \
#   --top-p 0.9
