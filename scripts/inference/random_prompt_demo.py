#!/usr/bin/env python3
"""Generate text from a randomly initialized miniLLaMA using a local tokenizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dummym.inference import generate_token_ids  # noqa: E402
from dummym.models.llama_like import (  # noqa: E402
    MiniLlamaConfig,
    MiniLlamaForCausalLM,
)


DEFAULT_TOKENIZER = Path("/diff/models/Qwen/Qwen3.5-2B/tokenizer.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a prompt through a randomly initialized miniLLaMA. The output "
            "is expected to be meaningless; this only validates the pipeline."
        )
    )
    parser.add_argument("prompt", nargs="?", default="你好，请介绍一下你自己。")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer}")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    eos_token_id = tokenizer.token_to_id("<|im_end|>")
    pad_token_id = tokenizer.token_to_id("<|endoftext|>")

    encoded = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not encoded.ids:
        raise ValueError("the prompt produced no token IDs")

    # This deliberately tiny architecture keeps the large borrowed Qwen
    # vocabulary affordable. It is a pipeline smoke model, not a ladder model.
    config = MiniLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=176,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=max(
            128,
            len(encoded.ids) + args.max_new_tokens,
        ),
        tie_word_embeddings=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    model = MiniLlamaForCausalLM(config)

    # BF16 is a natural smoke-test dtype on H20; CPU remains FP32.
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    all_token_ids = generate_token_ids(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_token_id,
        generator=rng,
    )

    prompt_length = input_ids.shape[1]
    new_token_ids = all_token_ids[0, prompt_length:].tolist()
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(all_token_ids[0].tolist(), skip_special_tokens=False)

    print(f"device: {device}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"vocab_size: {vocab_size}")
    print(f"model_parameters: {model.num_parameters():,}")
    print(f"prompt_tokens: {prompt_length}")
    print(f"generated_tokens: {len(new_token_ids)}")
    print(f"prompt: {args.prompt}")
    print(f"generated_text: {generated_text!r}")
    print(f"full_text: {full_text!r}")
    print(f"generated_token_ids: {new_token_ids}")


if __name__ == "__main__":
    main()
