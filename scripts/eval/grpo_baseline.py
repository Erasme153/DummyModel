#!/usr/bin/env python3
"""Audit GSM8K and measure the SFT policy's verifiable-reward sampling signal."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM  # noqa: E402
from scripts.train import pretrain as single  # noqa: E402
from scripts.train.sft_ddp import normalized_prompt, parquet_path  # noqa: E402

NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
FINAL_ANSWER = re.compile(rf"####\s*({NUMBER})\s*\Z")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/raw/m09_gsm8k/main")
    parser.add_argument("--ultrafeedback-dir", type=Path,
                        default=ROOT / "data/raw/m09_ultrafeedback/data")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs/m09_213m_sft/checkpoint.pt")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--split", choices=("holdout", "test"), default="holdout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--holdout-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-questions", type=int, default=0,
                        help="0 评测完整划分；正数只用于短程检查")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if any(getattr(args, name) <= 0 for name in
           ("holdout_size", "group_size", "max_new_tokens", "cpu_threads")):
        parser.error("holdout-size/group-size/max-new-tokens/cpu-threads 必须为正数")
    if args.max_questions < 0 or args.seed < 0 or \
            not math.isfinite(args.temperature) or args.temperature <= 0 or \
            args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("max-questions/seed/temperature/top-k/top-p 超出范围")
    if not args.audit_only and args.output is None:
        parser.error("基线评测必须指定 --output")
    return args


def final_number(text: str) -> Decimal | None:
    match = FINAL_ANSWER.search(text.strip())
    if match is None:
        return None
    return Decimal(match.group(1).replace(",", ""))


def prompt_ids(question: str, tokenizer: Tokenizer, config: MiniLlamaConfig):
    prompt = f"[INST] {question}\nSolve the problem. End your answer with #### <number>. [/INST]"
    return (config.bos_token_id, *tokenizer.encode(prompt, add_special_tokens=False).ids)


def load_data(data_dir: Path, ultrafeedback_dir: Path | None, tokenizer: Tokenizer,
              config: MiniLlamaConfig, *, holdout_size: int, max_new_tokens: int,
              seed: int, train_only: bool = False):
    paths = {name: parquet_path(data_dir, name) for name in ("train", "test")}
    raw = {"train": pq.read_table(paths["train"], columns=["question", "answer"]).to_pylist()}
    if train_only:
        test_questions = pq.read_table(paths["test"], columns=["question"])["question"].to_pylist()
    else:
        raw["test"] = pq.read_table(paths["test"], columns=["question", "answer"]).to_pylist()
        test_questions = [row["question"] for row in raw["test"]]
    test_keys = {normalized_prompt(question) for question in test_questions}
    ultra_keys = set()
    ultra_paths = {}
    if ultrafeedback_dir is not None:
        ultra_paths = {name: parquet_path(ultrafeedback_dir, name)
                       for name in ("train_sft", "test_prefs")}
        for path in ultra_paths.values():
            ultra_keys.update(normalized_prompt(value) for value in
                              pq.read_table(path, columns=["prompt"])["prompt"].to_pylist())
    if config.bos_token_id != tokenizer.token_to_id("<s>") or \
            config.eos_token_id != tokenizer.token_to_id("</s>"):
        raise ValueError("checkpoint 与 tokenizer 的 BOS/EOS 不一致")

    cleaned, counts = {}, {}
    for split in raw:
        seen = set()
        examples = []
        stat = {"raw": len(raw[split]), "train_test_overlap": 0,
                "ultrafeedback_overlap": 0, "duplicate_question": 0,
                "invalid_gold": 0, "prompt_too_long": 0}
        for source_index, row in enumerate(raw[split]):
            key = normalized_prompt(row["question"])
            if split == "train" and key in test_keys:
                stat["train_test_overlap"] += 1
                continue
            if key in ultra_keys:
                stat["ultrafeedback_overlap"] += 1
                # 留出集和官方测试都必须没有已知 SFT/偏好 prompt 重叠。
                continue
            if key in seen:
                stat["duplicate_question"] += 1
                continue
            answer = final_number(row["answer"])
            if answer is None:
                stat["invalid_gold"] += 1
                continue
            ids = prompt_ids(row["question"], tokenizer, config)
            if len(ids) + max_new_tokens > config.max_position_embeddings:
                stat["prompt_too_long"] += 1
                continue
            seen.add(key)
            examples.append({"source_index": source_index, "question": row["question"],
                             "answer": str(answer), "solution": row["answer"],
                             "prompt_ids": ids})
        stat["kept"] = len(examples)
        cleaned[split] = examples
        counts[split] = stat
    if holdout_size >= len(cleaned["train"]) or (not train_only and not cleaned["test"]):
        raise ValueError("GSM8K 训练/测试不足，无法建立独立留出集")
    order = torch.randperm(len(cleaned["train"]),
                           generator=torch.Generator().manual_seed(seed)).tolist()
    held_out = set(order[:holdout_size])
    holdout = [row for index, row in enumerate(cleaned["train"]) if index in held_out]
    train = [row for index, row in enumerate(cleaned["train"]) if index not in held_out]
    hashed_paths = {**paths, **ultra_paths}
    if train_only:
        del hashed_paths["test"]
    audit = {"splits": counts, "train_after_holdout": len(train),
             "holdout": len(holdout),
             "holdout_source_indices_sha256": hashlib.sha256(
                 json.dumps(sorted(row["source_index"] for row in holdout)).encode()).hexdigest(),
             "source_sha256": {name: single.file_sha256(path)
                               for name, path in hashed_paths.items()},
             "seed": seed, "max_new_tokens": max_new_tokens}
    if train_only:
        audit["test_questions_sha256"] = hashlib.sha256(json.dumps(
            test_questions, ensure_ascii=False).encode()).hexdigest()
    return {"train": train, "holdout": holdout, "test": cleaned.get("test", [])}, audit


@torch.inference_mode()
def generate_group(model, ids, tokenizer, *, group_size, max_new_tokens, temperature,
                   top_k, top_p, eos, generator, device, precision,
                   return_token_ids=False):
    generated = torch.tensor([ids] * group_size, dtype=torch.long, device=device)
    done = torch.zeros(group_size, dtype=torch.bool, device=device)
    lengths = torch.zeros(group_size, dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with single.precision_context(device, precision):
            logits = model(generated).logits[:, -1, :].float()
        logits = logits / temperature
        if top_k:
            values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
            filtered = torch.full_like(logits, -torch.inf)
            logits = filtered.scatter(1, indices, values)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        if top_p < 1:
            remove = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        sampled = sorted_indices.gather(1, torch.multinomial(
            probabilities, num_samples=1, generator=generator)).squeeze(1)
        sampled = torch.where(done, eos, sampled)
        lengths += (~done).long()
        done |= sampled == eos
        generated = torch.cat((generated, sampled[:, None]), dim=1)
        if done.all().item():
            break
    texts = []
    completions = []
    for row, length in zip(generated.tolist(), lengths.tolist()):
        completion = row[len(ids):len(ids) + length]
        completions.append(tuple(completion))
        if completion and completion[-1] == eos:
            completion = completion[:-1]
        texts.append(tokenizer.decode(completion, skip_special_tokens=True))
    if return_token_ids:
        return texts, lengths.tolist(), completions
    return texts, lengths.tolist()


def run(args):
    torch.set_num_threads(args.cpu_threads)
    if not args.audit_only and args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device 仅支持 cpu/cuda")
    if args.precision == "bf16" and (device.type != "cuda" or
                                     not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 需要支持该精度的 CUDA GPU")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True,
                            mmap=True)
    checkpoint_sha = single.file_sha256(args.checkpoint)
    if checkpoint.get("format_version") not in (3, 6):
        raise ValueError("GRPO 评测需要 SFT v3 或 GRPO v6 checkpoint")
    config = MiniLlamaConfig.from_dict(checkpoint["model_config"])
    if config.num_experts:
        raise ValueError("当前 GRPO 基线只支持 dense 模型")
    tokenizer_path = Path(checkpoint["tokenizer_path"])
    tokenizer_sha = single.file_sha256(tokenizer_path)
    if tokenizer_sha != checkpoint["contract"]["data"]["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer 指纹不符")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("checkpoint tokenizer 与模型词表不符")
    datasets, audit = load_data(args.data_dir, args.ultrafeedback_dir, tokenizer,
                                config, holdout_size=args.holdout_size,
                                max_new_tokens=args.max_new_tokens, seed=args.seed)
    if args.audit_only:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return audit
    selected = datasets[args.split]
    if args.max_questions:
        selected = selected[:args.max_questions]
    model = MiniLlamaForCausalLM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    started = time.perf_counter()
    records = []
    for index, row in enumerate(selected):
        generator = torch.Generator(device=device).manual_seed(args.seed + row["source_index"])
        texts, lengths = generate_group(
            model, row["prompt_ids"], tokenizer, group_size=args.group_size,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, eos=config.eos_token_id,
            generator=generator, device=device, precision=args.precision)
        gold = Decimal(row["answer"])
        predictions = [final_number(text) for text in texts]
        rewards = [int(value == gold) if value is not None else 0
                   for value in predictions]
        records.append({"source_index": row["source_index"], "question": row["question"],
                        "gold": row["answer"], "rewards": rewards,
                        "predictions": [str(value) if value is not None else None
                                        for value in predictions],
                        "parsed": [value is not None for value in predictions],
                        "generated_lengths": lengths, "completions": texts})
        if (index + 1) % 16 == 0 or index + 1 == len(selected):
            print(f"{index + 1}/{len(selected)} questions", flush=True)
    groups = len(records)
    samples = groups * args.group_size
    all_wrong = sum(not any(row["rewards"]) for row in records)
    all_correct = sum(all(row["rewards"]) for row in records)
    mixed = groups - all_wrong - all_correct
    parsed_samples = sum(sum(row["parsed"]) for row in records)
    correct_samples = sum(sum(row["rewards"]) for row in records)
    result = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": checkpoint_sha,
              "tokenizer_sha256": tokenizer_sha, "audit": audit,
              "split": args.split, "evaluated_questions": groups,
              "group_size": args.group_size, "max_new_tokens": args.max_new_tokens,
              "temperature": args.temperature, "top_k": args.top_k,
              "top_p": args.top_p, "seed": args.seed, "precision": args.precision,
              "metrics": {"parsed_samples": parsed_samples,
                          "parse_rate": parsed_samples / samples,
                          "correct_samples": correct_samples,
                          "sample_accuracy": correct_samples / samples,
                          "samples": samples,
                          "mean_generated_tokens": sum(
                              sum(row["generated_lengths"]) for row in records) / samples,
                          "first_sample_correct": sum(row["rewards"][0] for row in records),
                          "pass_at_1": sum(row["rewards"][0] for row in records) / groups,
                          "pass_at_k": (groups - all_wrong) / groups,
                          "all_wrong_groups": all_wrong, "all_correct_groups": all_correct,
                          "mixed_groups": mixed, "mixed_group_rate": mixed / groups},
              "elapsed_seconds": round(time.perf_counter() - started, 2),
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run(parse_args())
