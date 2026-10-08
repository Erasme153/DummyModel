"""M9 SFT 数据隔离、回复监督与恢复一致性。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer, models, pre_tokenizers

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from scripts.eval.base_eval import load_model
from scripts.eval import grpo_baseline as baseline
from scripts.train import pretrain as single
from scripts.train import pretrain_ddp as ddp
from scripts.train import sft_ddp as sft


@pytest.fixture
def tiny_sft(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "answer": 3, "one": 4, "two": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    def row(prompt, answer, prompt_id):
        return {"prompt": prompt, "prompt_id": prompt_id,
                "chosen": [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": answer}]}

    train = [row("Question one", "answer one", "1"),
             row("QUESTION   ONE", "answer two", "2"),  # 规范化后重复
             row("Leak prompt", "answer", "3"),         # 与测试集重复
             row("Question two", "answer two", "4"),
             row("Question three", "answer one", "5"),
             row("Question four", "answer two", "6")]
    test = [row("LEAK PROMPT", "answer", "7")]
    prefs = test + [row("Other prompt", "answer", "8")]
    for name, records in (("train_sft", train), ("test_sft", test), ("test_prefs", prefs)):
        pq.write_table(pa.Table.from_pylist(records), data / f"{name}-00000.parquet")
    config = MiniLlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.1)
    torch.manual_seed(11)
    model = MiniLlamaForCausalLM(config)
    checkpoint = tmp_path / "base.pt"
    torch.save({"format_version": 2, "model_config": config.to_dict(),
                "model_state_dict": model.state_dict(),
                "contract": {"world_size": 1,
                             "data": {"tokenizer_sha256": single.file_sha256(tokenizer_path)}},
                "tokenizer_path": str(tokenizer_path),
                "rng_states": [single.rng_state(torch.device("cpu"))]}, checkpoint)
    return data, tokenizer, config, checkpoint


def test_prompt_dedup_and_assistant_only_labels(tiny_sft):
    data, tokenizer, config, _ = tiny_sft
    examples, audit = sft.load_examples(data, tokenizer, config, 32)
    assert audit["train_sft"]["test_overlap"] == 1
    assert audit["train_sft"]["duplicate_prompt"] == 1
    assert audit["train_sft"]["kept"] == 4
    assert audit["test_sft_subset_of_test_prefs"] is True
    ids, labels, count = examples["train_sft"][0]
    assert ids[0] == config.bos_token_id
    assert labels[:len(ids) - count] == (-100,) * (len(ids) - count)
    assert labels[-count:] == ids[-count:]
    assert labels[-1] == config.eos_token_id
    batch_ids, batch_labels, mask = sft.collate(examples["train_sft"][:2],
                                                torch.device("cpu"), config.eos_token_id)
    assert torch.all(batch_labels[~mask] == -100)
    assert batch_ids.shape == batch_labels.shape == mask.shape


def test_gsm8k_sft_uses_baseline_split_and_v3_source(tiny_sft, tmp_path):
    _, tokenizer, config, checkpoint = tiny_sft
    data = tmp_path / "gsm8k"
    data.mkdir()
    train = [{"question": f"Question {index}", "answer": f"work\n#### {index}"}
             for index in range(6)]
    train.append({"question": "Ultra question", "answer": "work\n#### 9"})
    pq.write_table(pa.Table.from_pylist(train), data / "train-00000.parquet")
    pq.write_table(pa.Table.from_pylist([{"question": "Official test", "answer": "SECRET"}]),
                   data / "test-00000.parquet")
    ultra = tmp_path / "ultra"
    ultra.mkdir()
    for split in ("train_sft", "test_prefs"):
        pq.write_table(pa.Table.from_pylist([{"prompt": "Ultra question"}]),
                       ultra / f"{split}-00000.parquet")
    source = torch.load(checkpoint, weights_only=True)
    source["format_version"] = 3
    v3_checkpoint = tmp_path / "sft_source.pt"
    torch.save(source, v3_checkpoint)

    expected, expected_audit = baseline.load_data(
        data, ultra, tokenizer, config, holdout_size=2, max_new_tokens=4,
        seed=2026, train_only=True)
    examples, audit = sft.load_gsm8k_examples(
        data, ultra, tokenizer, config, 32, holdout_size=2, split_max_new_tokens=4)
    assert (len(examples["train_sft"]), len(examples["test_sft"])) == (4, 2)
    assert audit["holdout_source_indices_sha256"] == expected_audit["holdout_source_indices_sha256"]
    first = examples["train_sft"][0]
    assert first[0][:len(expected["train"][0]["prompt_ids"])] == expected["train"][0]["prompt_ids"]
    assert first[1][:len(expected["train"][0]["prompt_ids"])] == \
        (-100,) * len(expected["train"][0]["prompt_ids"])
    assert first[1][-1] == config.eos_token_id

    args = SimpleNamespace(dataset="gsm8k", raw_data_dir=data, ultrafeedback_dir=ultra,
                           holdout_size=2, split_max_new_tokens=4,
                           init_checkpoint=v3_checkpoint, output_dir=tmp_path / "math_sft",
                           resume=None, audit_only=False, precision="fp32", max_length=32,
                           batch_size=1, grad_accum_steps=1, epochs=1, learning_rate=1e-3,
                           min_lr_ratio=0.1, warmup_ratio=0, weight_decay=0.1,
                           beta1=0.9, beta2=0.95, max_grad_norm=1.0,
                           eval_every=1, save_every=1, log_every=1, seed=2026,
                           stop_after_steps=1)
    sft.run(args, ddp.Context(0, 1, torch.device("cpu")))
    saved = torch.load(args.output_dir / "checkpoint.pt", weights_only=True)
    assert saved["format_version"] == 3
    assert saved["contract"]["data"]["holdout_source_indices_sha256"] == \
        expected_audit["holdout_source_indices_sha256"]
    assert saved["progress"]["validation_supervised_tokens"] > 0
    resume_args = SimpleNamespace(**{**vars(args), "resume": args.output_dir / "checkpoint.pt",
                                     "output_dir": None, "stop_after_steps": 2})
    sft.run(resume_args, ddp.Context(0, 1, torch.device("cpu")))
    resumed = torch.load(args.output_dir / "checkpoint.pt", weights_only=True)
    assert resumed["progress"]["step"] == 2


def test_paused_sft_matches_uninterrupted_training(tiny_sft, tmp_path):
    data, _, _, checkpoint = tiny_sft
    common = dict(raw_data_dir=data, init_checkpoint=checkpoint, resume=None,
                  audit_only=False, precision="fp32", max_length=32, batch_size=1,
                  grad_accum_steps=2, epochs=1, learning_rate=1e-3,
                  min_lr_ratio=0.1, warmup_ratio=0, weight_decay=0.1,
                  beta1=0.9, beta2=0.95, max_grad_norm=1.0,
                  eval_every=1, save_every=1, log_every=1, seed=2026)
    ctx = ddp.Context(0, 1, torch.device("cpu"))
    paused = tmp_path / "paused"
    sft.run(SimpleNamespace(**common, output_dir=paused, stop_after_steps=1), ctx)
    resumed_args = {**common, "resume": paused / "checkpoint.pt"}
    sft.run(SimpleNamespace(**resumed_args, output_dir=None, stop_after_steps=None), ctx)
    full = tmp_path / "full"
    sft.run(SimpleNamespace(**common, output_dir=full, stop_after_steps=None), ctx)
    actual = torch.load(paused / "checkpoint.pt", weights_only=True)
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    assert actual["progress"] == expected["progress"]
    for key in actual["model_state_dict"]:
        torch.testing.assert_close(actual["model_state_dict"][key],
                                   expected["model_state_dict"][key], rtol=0, atol=0)


def _two_rank_worker(rank, rendezvous, data, source, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        ctx = ddp.Context(rank, 2, torch.device("cpu"))
        args = SimpleNamespace(raw_data_dir=Path(data), init_checkpoint=Path(source),
                               output_dir=Path(output), resume=None, audit_only=False,
                               precision="fp32", max_length=32, batch_size=1,
                               grad_accum_steps=1, epochs=1, learning_rate=1e-3,
                               min_lr_ratio=0.1, warmup_ratio=0, weight_decay=0.1,
                               beta1=0.9, beta2=0.95, max_grad_norm=1.0,
                               eval_every=1, save_every=1, log_every=1, seed=2026,
                               stop_after_steps=None)
        sft.run(args, ctx)
    finally:
        dist.destroy_process_group()


def test_two_rank_sft_completes(tiny_sft, tmp_path):
    data, _, _, checkpoint = tiny_sft
    source = torch.load(checkpoint, weights_only=True)
    source["contract"]["world_size"] = 2
    source["rng_states"] = [source["rng_states"][0], source["rng_states"][0]]
    two_rank_source = tmp_path / "base_two_rank.pt"
    torch.save(source, two_rank_source)
    output = tmp_path / "two_rank"
    mp.spawn(_two_rank_worker,
             args=(str(tmp_path / "rendezvous"), str(data), str(two_rank_source), str(output)),
             nprocs=2, join=True)
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["progress"]["step"] == 2
    assert saved["contract"]["world_size"] == 2
    evaluated_model, evaluated_config, _ = load_model(output / "checkpoint.pt", torch.device("cpu"))
    assert evaluated_model.num_parameters() > 0
    assert evaluated_config.to_dict() == saved["model_config"]
