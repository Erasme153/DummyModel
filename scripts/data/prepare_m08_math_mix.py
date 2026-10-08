#!/usr/bin/env python3
"""将独立 OpenWebMath 文档与 M8 FineWeb 序列按 1:1 打包为 M8 训练数据。

数学文档先与 M4/M8 的 documents.jsonl 做正文 SHA-256 去重，再按文档划分
训练/验证。仅数学训练流与 M8 FineWeb 训练流混合；验证集始终是独立数学文档。
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.prepare_fineweb import (  # noqa: E402
    PackedWriter, iter_dump_batches, load_excluded_documents, normalize_text,
    split_document, write_json_line,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-input-dir", type=Path,
                        default=Path("/diff/workspace/wyl/data/open-web-math"))
    parser.add_argument("--fineweb-data-dir", type=Path,
                        default=PROJECT_ROOT / "data/tokenized/m08_fineweb_100m")
    parser.add_argument("--exclude-documents", type=Path, action="append", required=True,
                        help="重复指定 M4 与 M8 的 documents.jsonl，按规范化正文哈希排除")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/tokenized/m08_math50_100m")
    parser.add_argument("--math-train-sequences", type=int, default=24_414)
    parser.add_argument("--fineweb-train-sequences", type=int, default=24_414)
    parser.add_argument("--validation-sequences", type=int, default=488)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128, help="读取 Parquet 时的文档数")
    parser.add_argument("--log-every", type=int, default=100, help="每多少个读取 batch 打印一次进度")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.math_train_sequences, args.fineweb_train_sequences,
           args.validation_sequences, args.batch_size, args.log_every) <= 0 or args.sequence_length < 2:
        parser.error("序列数和 batch-size 必须为正数，sequence-length 至少为 2")
    if args.math_train_sequences != args.fineweb_train_sequences:
        parser.error("本入口固定 1:1 混合，两个训练序列数必须相等")
    if not 0 < args.validation_fraction < 1 or not 0 < args.min_chars <= args.max_chars:
        parser.error("validation-fraction 或字符长度范围无效")
    if args.seed < 0 or len(args.exclude_documents) < 2:
        parser.error("seed 必须非负，且需要分别传入 M4 和 M8 文档索引")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_math_batches(files, root, seed, batch_size, sources):
    """跨分片轮询，避免约 50M token 预算只读到单个 OpenWebMath 分片。"""
    readers = deque(iter_dump_batches([path], root, f"{seed}:{path.name}", batch_size, sources)
                    for path in files)
    try:
        while readers:
            reader = readers.popleft()
            try:
                item = next(reader)
            except StopIteration:
                continue
            except BaseException:
                reader.close()
                raise
            readers.append(reader)
            yield item
    finally:
        for reader in readers:
            reader.close()


def prepare_math(args, files, tokenizer, excluded, output):
    eos = tokenizer.token_to_id("</s>")
    sources = {}
    counts = Counter()
    filtered = Counter()
    seen = set()
    math_train = output / ".math_train.bin"
    with ExitStack() as stack:
        writers = {
            "train": PackedWriter(stack.enter_context(math_train.open("wb")),
                                  args.math_train_sequences * args.sequence_length,
                                  args.sequence_length),
            "validation": PackedWriter(stack.enter_context((output / "validation.bin").open("wb")),
                                       args.validation_sequences * args.sequence_length,
                                       args.sequence_length),
        }
        documents = stack.enter_context((output / "math_documents.jsonl").open("w", encoding="utf-8"))
        batches = iter_math_batches(files, args.math_input_dir.resolve(), args.seed,
                                   args.batch_size, sources)
        stack.callback(batches.close)
        for batch_index, (source, rows) in enumerate(batches, 1):
            candidates = []
            for row in rows:
                counts["scanned_documents"] += 1
                text, reason = normalize_text(row["text"], args.min_chars, args.max_chars)
                if reason is not None:
                    filtered[reason] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest in excluded:
                    counts["excluded_documents"] += 1
                    continue
                if digest in seen:
                    counts["duplicate_documents"] += 1
                    continue
                seen.add(digest)
                split = split_document(digest, args.seed, args.validation_fraction)
                if writers[split].full:
                    counts["split_quota_skipped"] += 1
                    continue
                candidates.append((text, digest, split, row))
            encodings = tokenizer.encode_batch([item[0] for item in candidates], add_special_tokens=False)
            for (text, digest, split, row), encoding in zip(candidates, encodings):
                writer = writers[split]
                if writer.full:
                    counts["split_quota_skipped"] += 1
                    continue
                ids = encoding.ids + [eos]
                offset = writer.written_tokens + len(writer.pending)
                used = writer.add(ids)
                write_json_line(documents, {
                    "split": split, "sha256": digest.hex(), "source": source,
                    "url": row.get("url"), "characters": len(text),
                    "encoded_tokens_with_eos": len(ids), "used_tokens": used,
                    "math_token_offset": offset,
                })
                counts[f"{split}_documents"] += 1
            if batch_index % args.log_every == 0:
                print(f"batches={batch_index} scanned={counts['scanned_documents']:,} "
                      f"math_train={writers['train'].written_tokens:,}/"
                      f"{writers['train'].target_tokens:,} "
                      f"math_validation={writers['validation'].written_tokens:,}/"
                      f"{writers['validation'].target_tokens:,}", flush=True)
            if all(writer.full for writer in writers.values()):
                break
    if not all(writer.full for writer in writers.values()):
        raise RuntimeError("OpenWebMath 未达到训练或验证 token 预算；输出目录不含完成摘要")
    for source, metadata in sources.items():
        stat = (args.math_input_dir.resolve() / source).stat()
        if (stat.st_size, stat.st_mtime_ns) != (metadata["bytes"], metadata["mtime_ns"]):
            raise RuntimeError(f"准备期间原始分片发生变化：{source}")
    return math_train, {split: writer.stats() for split, writer in writers.items()}, {
        "files_read": sources, "counts": dict(counts), "filtered_documents": dict(filtered),
    }


def mix_train(args, fineweb_train, math_train, output):
    length = args.sequence_length
    fineweb = np.memmap(fineweb_train, dtype="<u2", mode="r").reshape(-1, length)
    math = np.memmap(math_train, dtype="<u2", mode="r").reshape(-1, length)
    if len(fineweb) < args.fineweb_train_sequences or len(math) != args.math_train_sequences:
        raise ValueError("来源训练序列数不符合混合预算")
    rng = np.random.default_rng(args.seed)
    fineweb_indices = rng.permutation(len(fineweb))[:args.fineweb_train_sequences]
    math_indices = rng.permutation(len(math))
    assignment = np.concatenate((np.zeros(len(fineweb_indices), dtype=np.uint8),
                                 np.ones(len(math_indices), dtype=np.uint8)))
    rng.shuffle(assignment)
    fineweb_cursor = math_cursor = 0
    with (output / "train.bin").open("wb") as handle:
        for start in range(0, len(assignment), 256):
            flags = assignment[start:start + 256]
            rows = np.empty((len(flags), length), dtype="<u2")
            fineweb_positions = np.flatnonzero(flags == 0)
            math_positions = np.flatnonzero(flags == 1)
            fw_end = fineweb_cursor + len(fineweb_positions)
            math_end = math_cursor + len(math_positions)
            rows[fineweb_positions] = fineweb[fineweb_indices[fineweb_cursor:fw_end]]
            rows[math_positions] = math[math_indices[math_cursor:math_end]]
            handle.write(rows.tobytes())
            fineweb_cursor, math_cursor = fw_end, math_end
    assert fineweb_cursor == args.fineweb_train_sequences
    assert math_cursor == args.math_train_sequences
    return {
        "fineweb_indices_sha256": hashlib.sha256(fineweb_indices.astype("<i8").tobytes()).hexdigest(),
        "math_indices_sha256": hashlib.sha256(math_indices.astype("<i8").tobytes()).hexdigest(),
        "source_assignment_sha256": hashlib.sha256(assignment.tobytes()).hexdigest(),
        "numpy_version": np.__version__,
    }


def main():
    args = parse_args()
    files = sorted(args.math_input_dir.resolve().glob("data/*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 {args.math_input_dir}/data/*.parquet")
    fineweb_dir = args.fineweb_data_dir.resolve()
    if (fineweb_dir / "documents.jsonl").resolve() not in {
            path.resolve() for path in args.exclude_documents}:
        raise ValueError("--exclude-documents 必须包含 M8 FineWeb 的 documents.jsonl")
    fineweb_summary = json.loads((fineweb_dir / "summary.json").read_text(encoding="utf-8"))
    if (fineweb_summary.get("status") != "complete"
            or fineweb_summary["format"]["dtype"] != "<u2"
            or fineweb_summary["format"]["sequence_length"] != args.sequence_length):
        raise ValueError("M8 FineWeb 数据必须完整，且为相同长度的 uint16 序列")
    fineweb_train = fineweb_dir / "train.bin"
    fineweb_train_stat = fineweb_train.stat()
    expected_bytes = fineweb_summary["splits"]["train"]["sequences"] * args.sequence_length * 2
    if fineweb_train_stat.st_size != expected_bytes:
        raise ValueError("M8 FineWeb train.bin 大小与摘要不一致")
    tokenizer_path = fineweb_dir / fineweb_summary["tokenizer"]["file"]
    tokenizer_hash = sha256(tokenizer_path)
    if tokenizer_hash != fineweb_summary["tokenizer"]["sha256"]:
        raise ValueError("M8 FineWeb tokenizer 与摘要哈希不一致")
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (tokenizer.get_vocab_size() != fineweb_summary["tokenizer"]["vocab_size"]
            or tokenizer.token_to_id("</s>") != fineweb_summary["tokenizer"]["eos_token_id"]):
        raise ValueError("M8 FineWeb tokenizer 的词表或 EOS 与摘要不一致")
    tokenizer.no_padding()
    tokenizer.no_truncation()
    excluded = set()
    exclusions = []
    for index in args.exclude_documents:
        hashes, index_hash = load_excluded_documents(index)
        excluded.update(hashes)
        exclusions.append({"path": str(index.resolve()), "sha256": index_hash,
                           "document_hashes": len(hashes)})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(tokenizer_path, output / "tokenizer.json")
    if sha256(output / "tokenizer.json") != tokenizer_hash:
        raise RuntimeError("复制后的 tokenizer 哈希不一致")
    math_train, math_stats, math_source = prepare_math(args, files, tokenizer, excluded, output)
    try:
        math_train_hash = sha256(math_train)
        mixing = mix_train(args, fineweb_train, math_train, output)
    finally:
        math_train.unlink(missing_ok=True)
    final_fineweb_stat = fineweb_train.stat()
    if (fineweb_train_stat.st_size, fineweb_train_stat.st_mtime_ns) != (
            final_fineweb_stat.st_size, final_fineweb_stat.st_mtime_ns):
        raise RuntimeError("准备期间 M8 FineWeb train.bin 发生变化")
    total_sequences = args.fineweb_train_sequences + args.math_train_sequences
    train_path, validation_path = output / "train.bin", output / "validation.bin"
    summary = {
        "status": "complete",
        "source": {
            "fineweb": {"data_dir": str(fineweb_dir), "train_sha256": sha256(fineweb_train),
                        "selected_sequences": args.fineweb_train_sequences,
                        "document_index": str(fineweb_dir / "documents.jsonl")},
            "openwebmath": {"input_dir": str(args.math_input_dir.resolve()),
                            "selected_sequences": args.math_train_sequences,
                            "packed_train_sha256": math_train_hash, **math_source},
        },
        "exclusion": {"indexes": exclusions, "unique_hashes": len(excluded)},
        "mixing": {"seed": args.seed, "method": "sample without replacement; shuffle 1:1 sequence assignment",
                   **mixing},
        "tokenizer": {"file": "tokenizer.json", "sha256": tokenizer_hash,
                      "vocab_size": tokenizer.get_vocab_size(),
                      "eos_token_id": tokenizer.token_to_id("</s>")},
        "format": {"dtype": "<u2", "sequence_length": args.sequence_length,
                   "packing": "independent math document streams; mixed by complete sequence"},
        "splits": {
            "train": {"sequences": total_sequences, "written_tokens": total_sequences * args.sequence_length,
                      "bytes": train_path.stat().st_size, "sha256": sha256(train_path),
                      "fineweb_sequences": args.fineweb_train_sequences,
                      "openwebmath_sequences": args.math_train_sequences},
            "validation": {**math_stats["validation"], "sha256": sha256(validation_path),
                           "source": "openwebmath"},
        },
        "math_documents_index": {"path": "math_documents.jsonl",
                                 "sha256": sha256(output / "math_documents.jsonl"),
                                 "scope": "OpenWebMath only; FineWeb document index remains in source data dir"},
    }
    if summary["splits"]["train"]["bytes"] != total_sequences * args.sequence_length * 2:
        raise RuntimeError("混合 train.bin 大小不符合序列预算")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"status": "complete", "train": summary["splits"]["train"],
                      "validation": summary["splits"]["validation"],
                      "math_counts": math_source["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
