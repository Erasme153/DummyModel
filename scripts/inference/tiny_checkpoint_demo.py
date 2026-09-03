#!/usr/bin/env python3
"""加载 tiny-overfit checkpoint，验证 loss，并执行一次贪心生成。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dummym.inference import generate_token_ids  # noqa: E402
from dummym.models.llama_like import (  # noqa: E402
    MiniLlamaConfig,
    MiniLlamaForCausalLM,
)
from scripts.train.tiny_overfit import (  # noqa: E402
    build_fixed_batch,
    next_token_accuracy,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "runs" / "m00_tiny_overfit" / "checkpoint.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and generate from a tiny-overfit checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--prompt",
        default="The small language model",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--num-sequences", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-loss", type=float, default=0.06)
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}"
        )

    # 只加载自己生成、来源可信的 .pt 文件。
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    # 使用 checkpoint 中保存的模型配置重建模型。
    config = MiniLlamaConfig.from_dict(
        checkpoint["model_config"]
    )
    model = MiniLlamaForCausalLM(config)

    # strict=True 保证没有缺少、多出或名称不匹配的参数。
    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    # 使用训练时完全相同的 Tokenizer 和语料。
    tokenizer_path = Path(checkpoint["tokenizer_path"])
    corpus_path = Path(checkpoint["corpus_path"])

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if config.eos_token_id is None:
        raise ValueError("checkpoint config does not contain eos_token_id")

    # 重建训练时反复使用的固定 batch。
    fixed_batch, document_count = build_fixed_batch(
        corpus_path,
        tokenizer,
        eos_token_id=config.eos_token_id,
        sequence_length=config.max_position_embeddings,
        num_sequences=args.num_sequences,
    )
    fixed_batch = fixed_batch.to(device)

    # 第一项测试：重新计算固定 batch 的 loss 和准确率。
    with torch.inference_mode():
        output = model(
            fixed_batch,
            labels=fixed_batch,
        )
        assert output.loss is not None

        accuracy = next_token_accuracy(
            output.logits,
            fixed_batch,
        )

    evaluated_loss = output.loss.item()
    evaluated_accuracy = accuracy.item()

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"load_result: {load_result}")
    print(f"checkpoint_step: {checkpoint['step']}")
    print(f"saved_loss: {checkpoint['final_loss']:.6f}")
    print(f"evaluated_loss: {evaluated_loss:.6f}")
    print(f"next_token_accuracy: {evaluated_accuracy:.4f}")
    print(f"documents: {document_count}")

    if evaluated_loss > args.max_loss:
        raise RuntimeError(
            f"loss check failed: "
            f"{evaluated_loss:.6f} > {args.max_loss:.6f}"
        )

    if evaluated_accuracy < args.min_accuracy:
        raise RuntimeError(
            f"accuracy check failed: "
            f"{evaluated_accuracy:.4f} < "
            f"{args.min_accuracy:.4f}"
        )

    # 第二项测试：使用训练语料的前缀执行贪心生成。
    prompt_ids = tokenizer.encode(
        args.prompt,
        add_special_tokens=False,
    ).ids

    if not prompt_ids:
        raise ValueError("prompt produced no token IDs")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_ids = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        eos_token_id=config.eos_token_id,
    )

    new_token_ids = generated_ids[
        0, len(prompt_ids):
    ].tolist()

    generated_text = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=False,
    )

    print(f"prompt: {args.prompt!r}")
    print(f"generated_text: {generated_text!r}")
    print("checkpoint test: PASS")


if __name__ == "__main__":
    main()