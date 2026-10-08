"""GSM8K 审计、可验证奖励与分组采样基线。"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from scripts.eval import grpo_baseline as baseline
from scripts.train import pretrain as single


@pytest.fixture
def tiny_gsm8k(tmp_path):
    data = tmp_path / "gsm8k" / "main"
    data.mkdir(parents=True)
    train = [
        {"question": "Question one", "answer": "work\n#### 1"},
        {"question": "QUESTION   ONE", "answer": "work\n#### 1"},
        {"question": "Test question", "answer": "work\n#### 2"},
        {"question": "Ultra question", "answer": "work\n#### 3"},
        {"question": "Question two", "answer": "work\n#### 2"},
        {"question": "Question three", "answer": "work\n#### 3"},
        {"question": "Question four", "answer": "work\n#### 4"},
        {"question": "Broken gold", "answer": "answer: four"},
    ]
    test = [{"question": "Test question", "answer": "work\n#### 2"}]
    for name, rows in (("train", train), ("test", test)):
        pq.write_table(pa.Table.from_pylist(rows), data / f"{name}-00000.parquet")
    ultra = tmp_path / "ultra"
    ultra.mkdir()
    for name, prompt in (("train_sft", "ULTRA QUESTION"),
                         ("test_prefs", "Other question")):
        pq.write_table(pa.Table.from_pylist([{"prompt": prompt}]),
                       ultra / f"{name}-00000.parquet")

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "####": 3,
             "1": 4, "2": 5, "3": 6, "4": 7}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = MiniLlamaConfig(vocab_size=len(vocab), hidden_size=16,
                             intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=64, bos_token_id=1,
                             eos_token_id=2)
    torch.manual_seed(7)
    model = MiniLlamaForCausalLM(config)
    checkpoint = tmp_path / "sft.pt"
    torch.save({"format_version": 3, "model_config": config.to_dict(),
                "model_state_dict": model.state_dict(),
                "contract": {"data": {"tokenizer_sha256": single.file_sha256(tokenizer_path)}},
                "tokenizer_path": str(tokenizer_path)}, checkpoint)
    return data, ultra, tokenizer, config, checkpoint


def test_final_number_requires_terminal_marker():
    assert baseline.final_number("reasoning\n#### 1,234.50\n") == Decimal("1234.50")
    assert baseline.final_number("#### -0.25") == Decimal("-0.25")
    assert baseline.final_number("reasoning 42") is None
    assert baseline.final_number("#### 42\nmore text") is None
    assert baseline.final_number("#### 4/2") is None


def test_audit_split_isolation_and_reproducibility(tiny_gsm8k):
    data, ultra, tokenizer, config, _ = tiny_gsm8k
    datasets, audit = baseline.load_data(data, ultra, tokenizer, config,
                                         holdout_size=2, max_new_tokens=4, seed=2026)
    assert audit["splits"]["train"] == {
        "raw": 8, "train_test_overlap": 1, "ultrafeedback_overlap": 1,
        "duplicate_question": 1, "invalid_gold": 1,
        "prompt_too_long": 0, "kept": 4}
    assert audit["splits"]["test"]["kept"] == 1
    assert audit["train_after_holdout"] == 2
    assert audit["holdout"] == 2
    assert not ({row["question"] for row in datasets["holdout"]} &
                {row["question"] for row in datasets["train"]})
    assert datasets["test"][0]["question"] == "Test question"
    _, second = baseline.load_data(data, ultra, tokenizer, config,
                                   holdout_size=2, max_new_tokens=4, seed=2026)
    assert audit == second


def test_train_only_reuses_split_without_reading_official_answers(tiny_gsm8k, monkeypatch):
    data, ultra, tokenizer, config, _ = tiny_gsm8k
    _, baseline_audit = baseline.load_data(data, ultra, tokenizer, config,
                                           holdout_size=2, max_new_tokens=4, seed=2026)
    original_read_table = baseline.pq.read_table

    def checked_read_table(path, *args, **kwargs):
        if Path(path).parent == data and Path(path).name.startswith("test-"):
            assert kwargs["columns"] == ["question"]
        return original_read_table(path, *args, **kwargs)

    monkeypatch.setattr(baseline.pq, "read_table", checked_read_table)
    datasets, audit = baseline.load_data(data, ultra, tokenizer, config,
                                         holdout_size=2, max_new_tokens=4,
                                         seed=2026, train_only=True)
    assert len(datasets["train"]) == 2
    assert len(datasets["holdout"]) == 2
    assert datasets["test"] == []
    assert audit["holdout_source_indices_sha256"] == baseline_audit["holdout_source_indices_sha256"]
    assert "test" not in audit["source_sha256"]


def test_baseline_sampling_output_is_deterministic(tiny_gsm8k, tmp_path):
    data, ultra, _, _, checkpoint = tiny_gsm8k
    common = dict(data_dir=data, ultrafeedback_dir=ultra, checkpoint=checkpoint,
                  audit_only=False, split="holdout", holdout_size=2, group_size=2,
                  max_new_tokens=4, max_questions=0, temperature=0.7,
                  top_k=4, top_p=0.95, seed=2026, device="cpu",
                  precision="fp32", cpu_threads=1)
    first = baseline.run(SimpleNamespace(**common, output=tmp_path / "first.json"))
    second = baseline.run(SimpleNamespace(**common, output=tmp_path / "second.json"))
    assert first["records"] == second["records"]
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["samples"] == 4
    assert first["metrics"]["mixed_groups"] + first["metrics"]["all_wrong_groups"] + \
        first["metrics"]["all_correct_groups"] == 2
    with pytest.raises(FileExistsError):
        baseline.run(SimpleNamespace(**common, output=tmp_path / "first.json"))
