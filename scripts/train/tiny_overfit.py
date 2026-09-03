#!/usr/bin/env python3
"""在一个固定的小 batch 上反复训练 miniLLaMA，验证最小训练链路。

这个实验的目标不是获得有用的语言模型，而是让模型记住少量样本。如果 loss
能够从随机初始化时的约 ``log(vocab_size)`` 明显下降并接近 0，就说明至少以下
环节可以连通：Tokenizer、EOS 拼接、定长 packing、causal LM loss、反向传播、
AdamW 参数更新、TensorBoard 日志以及 checkpoint 保存。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


# 脚本位于 scripts/train/，向上两级就是项目根目录。将 src/ 放入模块搜索路径，
# 这样没有执行 ``pip install -e .`` 时也能直接从仓库运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "tiny_corpus.txt"
DEFAULT_TOKENIZER = Path("/diff/models/mistralai/Mistral-7B-v0.1/tokenizer.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "m00_tiny_overfit"


def parse_args() -> argparse.Namespace:
    """定义命令行参数及这个教学实验的默认配置。"""

    parser = argparse.ArgumentParser(
        description="Repeatedly train miniLLaMA on one fixed batch until it memorizes it."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--num-sequences", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """解析训练设备，并在显式请求 CUDA 但不可用时尽早报错。"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def validate_args(args: argparse.Namespace) -> None:
    """在创建模型前检查参数和输入文件，避免训练中途才失败。"""

    positive_values = {
        "steps": args.steps,
        "sequence_length": args.sequence_length,
        "num_sequences": args.num_sequences,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "target_loss": args.target_loss,
        "log_every": args.log_every,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.corpus.is_file():
        raise FileNotFoundError(f"corpus not found: {args.corpus}")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer}")


def build_fixed_batch(
    corpus_path: Path,
    tokenizer: Tokenizer,
    *,
    eos_token_id: int,
    sequence_length: int,
    num_sequences: int,
) -> tuple[torch.Tensor, int]:
    """将逐行文本转换成一个会被重复使用的定长 batch。

    每个非空行视为一篇独立文档。文档先由 Tokenizer 编码，再追加 EOS，最后
    连接成一条 token stream。函数从这条流中取出 ``num_sequences *
    sequence_length`` 个 token，并 reshape 为 ``[B, T]``。

    如果语料过短，就重复整条 token stream；这只适合 tiny-overfit 正确性测试，
    正式预训练应使用真正的数据 packing/sharding 流水线。

    Returns:
        fixed_batch: dtype 为 ``torch.long``、形状为 ``[B, T]`` 的 token ID。
        document_count: 实际读取的非空文档行数。
    """

    # 空行不作为文档，避免为空文档单独追加 EOS。
    documents = [
        line.strip()
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not documents:
        raise ValueError("the corpus must contain at least one non-empty line")

    # add_special_tokens=False 防止 Tokenizer 自动添加 BOS/EOS；这里显式追加 EOS，
    # 让文档边界的处理清晰可见，也保证每一行恰好有一个结束标记。
    token_stream: list[int] = []
    for document in documents:
        token_stream.extend(tokenizer.encode(document, add_special_tokens=False).ids)
        token_stream.append(eos_token_id)

    required_tokens = sequence_length * num_sequences
    if len(token_stream) < required_tokens:
        # 向上取整计算至少需要重复几次，确保后续切片一定有足够 token。
        repeats = (required_tokens + len(token_stream) - 1) // len(token_stream)
        token_stream = token_stream * repeats

    # 只保留所需长度，然后将连续 token 流切成 B 个长度为 T 的训练序列。
    fixed_batch = torch.tensor(
        token_stream[:required_tokens],
        dtype=torch.long,
    ).view(num_sequences, sequence_length)
    return fixed_batch, len(documents)


def next_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算 causal LM 的 next-token accuracy。

    位置 ``t`` 的 logits 用来预测位置 ``t + 1`` 的 token，因此 logits 去掉最后
    一个位置，labels 去掉第一个位置，与模型内部计算 loss 的 shift 完全一致。
    """

    predictions = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    return (predictions == targets).float().mean()


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    # 固定随机种子，使模型初始化和同一硬件/软件栈上的训练轨迹尽量可复现。
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # 直接读取 Hugging Face Tokenizers 保存的 tokenizer.json，不加载任何预训练
    # 模型权重。本实验明确要求 32K 词表，并使用 Mistral 的 </s> 作为 EOS。
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size != 32_000:
        raise ValueError(f"expected a 32K tokenizer, found vocab_size={vocab_size}")
    eos_token_id = tokenizer.token_to_id("</s>")
    if eos_token_id is None:
        raise ValueError("the tokenizer does not define the expected </s> EOS token")

    fixed_batch, document_count = build_fixed_batch(
        args.corpus,
        tokenizer,
        eos_token_id=eos_token_id,
        sequence_length=args.sequence_length,
        num_sequences=args.num_sequences,
    )
    fixed_batch = fixed_batch.to(device)

    # 使用约 2.1M 参数的极小 LLaMA-like 模型，以便快速验证训练正确性。这里直接
    # 写出模型维度，避免 tiny-overfit 被额外的配置管理逻辑遮蔽。
    config = MiniLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=176,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=args.sequence_length,
        attention_dropout=0.0,
        # 输入 Embedding 和输出 LM Head 共享同一个权重矩阵，减少参数量。
        tie_word_embeddings=True,
        bos_token_id=tokenizer.token_to_id("<s>"),
        eos_token_id=eos_token_id,
    )
    model = MiniLlamaForCausalLM(config).to(device)
    # tiny-overfit 只验证能否记住固定 batch，因此使用最小 AdamW 配置，不引入
    # scheduler、warmup、梯度累积或混合精度等额外变量。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # SummaryWriter 会在 tensorboard/ 下创建 events.out.tfevents.* 事件文件。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    print(f"device: {device}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"corpus: {args.corpus} ({document_count} documents)")
    print(f"fixed_batch: {tuple(fixed_batch.shape)}")
    print(f"model_parameters: {model.num_parameters():,}")
    print(f"tensorboard: {tensorboard_dir}")

    model.train()
    started_at = time.perf_counter()
    initial_loss: float | None = None
    final_loss = float("nan")
    final_accuracy = float("nan")
    completed_steps = 0

    try:
        for step in range(1, args.steps + 1):
            # set_to_none=True 不把旧梯度写成全 0，而是释放其引用；下一次 backward
            # 会创建新梯度，通常能减少内存写入。
            optimizer.zero_grad(set_to_none=True)

            # 输入和 labels 使用同一个固定 batch。模型内部会执行 causal shift：
            # logits[:, :-1] 预测 labels[:, 1:]，不会让 token 预测它自己。
            output = model(fixed_batch, labels=fixed_batch)
            assert output.loss is not None
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")

            # 反向传播得到梯度。clip_grad_norm_ 返回裁剪前的总梯度范数，并将超过
            # 阈值的梯度按比例缩小；随后 AdamW 才真正更新参数。
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # accuracy 只是便于理解“是否已经记住”的辅助指标，不参与反向传播。
            # 这里复用本 step 参数更新前的 logits，避免额外再做一次前向计算。
            with torch.no_grad():
                accuracy = next_token_accuracy(output.logits, fixed_batch)

            final_loss = loss.item()
            final_accuracy = accuracy.item()
            completed_steps = step
            if initial_loss is None:
                initial_loss = final_loss

            # 每一步都写 TensorBoard；--log-every 只控制终端打印频率。
            writer.add_scalar("train/loss", final_loss, step)
            writer.add_scalar("train/next_token_accuracy", final_accuracy, step)
            writer.add_scalar("train/gradient_norm", float(grad_norm), step)
            writer.add_scalar("train/learning_rate", args.learning_rate, step)

            if step == 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"step={step:04d} loss={final_loss:.6f} "
                    f"accuracy={final_accuracy:.4f} grad_norm={float(grad_norm):.4f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            # 达到预先设定的近零 loss 就提前结束，无需跑满 --steps。
            if final_loss <= args.target_loss:
                print(f"target loss {args.target_loss} reached at step {step}")
                break
    finally:
        # 即使训练抛出异常也关闭 writer，尽量把已记录的 event 刷到磁盘。
        writer.flush()
        writer.close()

    elapsed_seconds = time.perf_counter() - started_at
    checkpoint_path = args.output_dir / "checkpoint.pt"
    # 保存模型和优化器状态以及最关键的运行元数据。这个 checkpoint 足够用于
    # 教学实验的加载/继续训练，但还不是正式预训练所需的完整分布式恢复格式。
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": completed_steps,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "final_next_token_accuracy": final_accuracy,
            "seed": args.seed,
            "tokenizer_path": str(args.tokenizer),
            "corpus_path": str(args.corpus),
        },
        checkpoint_path,
    )

    # JSON 摘要便于不加载二进制 checkpoint 就快速查看本次运行结果。
    summary = {
        "device": str(device),
        "documents": document_count,
        "fixed_batch_shape": list(fixed_batch.shape),
        "model_parameters": model.num_parameters(),
        "completed_steps": completed_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_next_token_accuracy": final_accuracy,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(checkpoint_path),
        "tensorboard_log_dir": str(tensorboard_dir),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {summary_path}")
    # 将实验验收条件写成程序断言：命令成功退出就代表 loss 确实下降且达到目标。
    if initial_loss is None or final_loss >= initial_loss:
        raise RuntimeError("loss did not decrease")
    if final_loss > args.target_loss:
        raise RuntimeError(
            f"target loss was not reached: final={final_loss:.6f}, target={args.target_loss}"
        )


if __name__ == "__main__":
    main()
