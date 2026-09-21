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


def run_prepare(monkeypatch, corpus, output, tokens=512):
    root, tokenizer = corpus
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--input-dir", str(root),
                        "--tokenizer", str(tokenizer), "--output-dir", str(output),
                        "--train-tokens", str(tokens), "--validation-tokens", str(tokens),
                        "--sequence-length", "16", "--validation-fraction", "0.5", "--batch-size", "4"])
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


def test_invalid_parquet_is_not_silently_skipped(monkeypatch, corpus, tmp_path):
    root, _ = corpus
    for path in root.rglob("*.parquet"):
        pq.write_table(pa.table({"wrong_column": ["example"]}), path)
    output = tmp_path / "invalid"
    with pytest.raises(RuntimeError, match="读取 Parquet 失败"):
        run_prepare(monkeypatch, corpus, output)
    assert not (output / "summary.json").exists()
