"""数据准备的回归检查：隔离文档、保持 token 顺序、拒绝覆盖以及预算不足报错。"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/data/prepare_fineweb.py"
spec = importlib.util.spec_from_file_location("prepare_fineweb", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
MIX_SCRIPT = SCRIPT.with_name("prepare_m08_math_mix.py")
mix_spec = importlib.util.spec_from_file_location("prepare_m08_math_mix", MIX_SCRIPT)
mix = importlib.util.module_from_spec(mix_spec)
mix_spec.loader.exec_module(mix)


def test_normalization_preserves_paragraphs_and_rejects_bad_text():
    text, reason = prepare.normalize_text("  Cafe\u0301\tlesson\r\n\r\n\r\n Second paragraph  ", 1, 200)
    assert reason is None
    assert text == "Café lesson\n\nSecond paragraph"
    assert prepare.normalize_text(None, 1, 200)[1] == "not_text"
    assert prepare.normalize_text("broken\x00text", 1, 200)[1] == "control_character"
    assert prepare.normalize_text("bad \ufffd decoding", 1, 200)[1] == "replacement_character"
    assert prepare.normalize_text("1234567890", 1, 200)[1] == "low_letter_fraction"


def test_packing_keeps_eos_and_stream_order_without_repeating_or_padding():
    train_file, validation_file = io.BytesIO(), io.BytesIO()
    train = prepare.PackedWriter(train_file, requested_tokens=10, sequence_length=4)
    validation = prepare.PackedWriter(validation_file, requested_tokens=4, sequence_length=4)
    assert train.add([10, 11, 2]) == 3  # 暂存尾巴，不能补 padding 或重复文本。
    assert train_file.getvalue() == b""
    validation.add([20, 21, 22, 2])
    assert train.add([12, 13, 14, 15, 16, 2]) == 5  # 预算向下对齐到 8。
    assert train.full
    assert train.add([99, 2]) == 0
    assert np.frombuffer(train_file.getvalue(), dtype="<u2").tolist() == [10, 11, 2, 12, 13, 14, 15, 16]
    assert np.frombuffer(validation_file.getvalue(), dtype="<u2").tolist() == [20, 21, 22, 2]
    assert train.stats()["truncated_tokens_at_budget"] == 1


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "raw"
    # 两个抓取批次包含完全相同的正文，只有尾部空白不同，必须被规范化后去重。
    texts = [f"English educational article number {index} discusses science and learning. " * 4
             for index in range(60)]
    for dump in ("CC-MAIN-2013-20", "CC-MAIN-2014-10"):
        directory = root / "data" / dump
        directory.mkdir(parents=True)
        pq.write_table(pa.table({"text": [text + "\n " for text in texts],
                                 "id": [str(index) for index in range(60)]}),
                       directory / "part.parquet", row_group_size=5)
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"word{index}": index + 3 for index in range(31_997)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return root, path


def run_prepare(monkeypatch, corpus, output, tokens=512, exclude_documents=None):
    root, tokenizer = corpus
    argv = [str(SCRIPT), "--input-dir", str(root),
            "--tokenizer", str(tokenizer), "--output-dir", str(output),
            "--train-tokens", str(tokens), "--validation-tokens", str(tokens),
            "--sequence-length", "16", "--validation-fraction", "0.5", "--batch-size", "4"]
    if exclude_documents is not None:
        argv.extend(("--exclude-documents", str(exclude_documents)))
    monkeypatch.setattr(sys, "argv", argv)
    prepare.main()


def test_pipeline_is_reproducible_disjoint_and_uses_multiple_dumps(monkeypatch, corpus, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_prepare(monkeypatch, corpus, first)
    run_prepare(monkeypatch, corpus, second)
    for name in ("train.bin", "validation.bin", "documents.jsonl", "preview.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    summary = json.loads((first / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert len(summary["source"]["selected_by_source"]) == 2
    docs = [json.loads(line) for line in (first / "documents.jsonl").read_text().splitlines()]
    hashes = [doc["sha256"] for doc in docs]
    assert len(set(hashes)) == len(hashes)  # 同时保证 split 内及跨 split 没有完全重复。
    for split in ("train", "validation"):
        data = np.fromfile(first / f"{split}.bin", dtype="<u2")
        assert data.size == 512
        assert data.reshape(-1, 16).shape == (32, 16)
        for doc in (doc for doc in docs if doc["split"] == split):
            if doc["used_tokens"] == doc["encoded_tokens_with_eos"]:
                assert data[doc["token_offset"] + doc["used_tokens"] - 1] == 2
    with pytest.raises(FileExistsError):
        run_prepare(monkeypatch, corpus, first)


def test_insufficient_corpus_is_reported_and_duplicates_are_removed(monkeypatch, corpus, tmp_path):
    output = tmp_path / "insufficient"
    with pytest.raises(RuntimeError, match="未达到预算"):
        run_prepare(monkeypatch, corpus, output, tokens=100_000)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "insufficient_data"
    assert summary["counts"]["duplicate_documents"] == 60
    assert sum(split["documents"] for split in summary["splits"].values()) == 60


def test_prior_document_index_is_excluded_before_packing(monkeypatch, corpus, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_prepare(monkeypatch, corpus, first, tokens=128)
    index = first / "documents.jsonl"
    run_prepare(monkeypatch, corpus, second, tokens=128, exclude_documents=index)
    old = {json.loads(line)["sha256"] for line in index.read_text().splitlines()}
    new = {json.loads(line)["sha256"] for line in (second / "documents.jsonl").read_text().splitlines()}
    summary = json.loads((second / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert old.isdisjoint(new)
    assert summary["exclusion"]["unique_hashes"] == len(old)
    assert summary["exclusion"]["skipped_documents"] > 0


def test_invalid_parquet_is_not_silently_skipped(monkeypatch, corpus, tmp_path):
    root, _ = corpus
    for path in root.rglob("*.parquet"):
        pq.write_table(pa.table({"wrong_column": ["example"]}), path)
    output = tmp_path / "invalid"
    with pytest.raises(RuntimeError, match="读取 Parquet 失败"):
        run_prepare(monkeypatch, corpus, output)
    assert not (output / "summary.json").exists()


def test_m08_math_mix_is_reproducible_and_excludes_both_document_indexes(monkeypatch, corpus, tmp_path):
    fineweb = tmp_path / "fineweb"
    run_prepare(monkeypatch, corpus, fineweb)
    overlap = json.loads((fineweb / "preview.jsonl").read_text().splitlines()[0])["text_preview"]
    math_texts = [f"word{index} explains mathematics and algebra with several detailed examples. " * 5
                  for index in range(100)]
    math_root = tmp_path / "openwebmath"
    (math_root / "data").mkdir(parents=True)
    texts = []
    for start in range(0, len(math_texts), 10):
        texts.extend((overlap, math_texts[0], *math_texts[start:start + 10]))
    for shard, selected in enumerate((texts[:60], texts[60:])):
        pq.write_table(pa.table({"text": selected,
                                 "url": [f"https://math.example/{shard}/{index}"
                                         for index in range(len(selected))]}),
                       math_root / f"data/train-{shard}.parquet", row_group_size=12)
    other_exclusion = tmp_path / "m04_documents.jsonl"
    other_text, _ = prepare.normalize_text(math_texts[0], 200, 100_000)
    import hashlib
    other_exclusion.write_text(json.dumps({"sha256": hashlib.sha256(other_text.encode()).hexdigest()}) + "\n")

    def run_mix(output):
        monkeypatch.setattr(sys, "argv", [str(MIX_SCRIPT), "--math-input-dir", str(math_root),
                                        "--fineweb-data-dir", str(fineweb),
                                        "--exclude-documents", str(fineweb / "documents.jsonl"),
                                        "--exclude-documents", str(other_exclusion),
                                        "--output-dir", str(output),
                                        "--fineweb-train-sequences", "24", "--math-train-sequences", "24",
                                        "--validation-sequences", "16", "--sequence-length", "16",
                                        "--validation-fraction", "0.5", "--batch-size", "12"])
        mix.main()

    first, second = tmp_path / "mixed_first", tmp_path / "mixed_second"
    run_mix(first)
    run_mix(second)
    summary = json.loads((first / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["splits"]["train"]["fineweb_sequences"] == 24
    assert summary["splits"]["train"]["openwebmath_sequences"] == 24
    assert summary["splits"]["train"]["written_tokens"] == 768
    assert summary["splits"]["validation"]["written_tokens"] == 256
    assert summary["source"]["openwebmath"]["counts"]["excluded_documents"] >= 2
    assert len(summary["source"]["openwebmath"]["files_read"]) == 2
    excluded = {json.loads(line)["sha256"] for line in (fineweb / "documents.jsonl").read_text().splitlines()}
    excluded.add(json.loads(other_exclusion.read_text())["sha256"])
    math_docs = [json.loads(line) for line in (first / "math_documents.jsonl").read_text().splitlines()]
    assert {doc["sha256"] for doc in math_docs}.isdisjoint(excluded)
    assert len({doc["sha256"] for doc in math_docs}) == len(math_docs)
    for name in ("train.bin", "validation.bin", "math_documents.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert np.fromfile(first / "train.bin", dtype="<u2").reshape(48, 16).shape == (48, 16)
    with pytest.raises(FileExistsError):
        run_mix(first)
