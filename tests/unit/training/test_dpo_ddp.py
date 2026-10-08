"""DPO 偏好数据隔离、目标函数与断点恢复。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer, models, pre_tokenizers

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from scripts.eval.base_eval import load_model
from scripts.train import dpo_ddp as dpo
from scripts.train import pretrain as single
from scripts.train import pretrain_ddp as ddp


@pytest.fixture
def tiny_dpo(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "good": 3, "bad": 4,
             "one": 5, "two": 6, "yes": 7, "no": 8}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    def row(prompt, chosen, rejected, prompt_id, chosen_score=8.0, rejected_score=4.0):
        def messages(answer):
            return [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}]
        return {"prompt": prompt, "prompt_id": prompt_id,
                "chosen": messages(chosen), "rejected": messages(rejected),
                "score_chosen": chosen_score, "score_rejected": rejected_score}

    train = [row("Question one", "good one", "bad one", "1"),
             row("QUESTION   ONE", "good two", "bad two", "2"),
             row("Preference holdout", "good", "bad", "3"),
             row("Tied question", "good", "bad", "4", 5, 5),
             row("Same answer", "good one", "GOOD   ONE", "5"),
             row("Long answer", "good " * 40, "bad", "6"),
             row("Question two", "good two", "bad two", "7"),
             row("Question three", "yes", "no", "8"),
             row("Question four", "good", "bad", "9")]
    sft_test = [row("SFT holdout", "good", "bad", "10")]
    prefs_test = sft_test + [row("Preference holdout", "good", "bad", "11"),
                             row("Extra holdout", "yes", "no", "12")]
    for name, records in (("train_prefs", train), ("test_sft", sft_test),
                          ("test_prefs", prefs_test)):
        pq.write_table(pa.Table.from_pylist(records), data / f"{name}-00000.parquet")
    config = MiniLlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.1)
    torch.manual_seed(11)
    model = MiniLlamaForCausalLM(config)
    checkpoint = tmp_path / "sft.pt"
    torch.save({"format_version": 3, "model_config": config.to_dict(),
                "model_state_dict": model.state_dict(),
                "contract": {"world_size": 1,
                             "data": {"tokenizer_sha256": single.file_sha256(tokenizer_path)}},
                "tokenizer_path": str(tokenizer_path),
                "rng_states": [single.rng_state(torch.device("cpu"))]}, checkpoint)
    return data, tokenizer, config, checkpoint


def test_audit_and_prompt_mask(tiny_dpo):
    data, tokenizer, config, _ = tiny_dpo
    examples, audit = dpo.load_examples(data, tokenizer, config, 32)
    assert audit["train_prefs"]["train_test_overlap"] == 1
    assert audit["train_prefs"]["duplicate_prompt"] == 1
    assert audit["train_prefs"]["tie_or_reverse"] == 1
    assert audit["train_prefs"]["invalid_or_identical"] == 1
    assert audit["train_prefs"]["overlength_or_empty"] == 1
    assert audit["train_prefs"]["kept"] == 4
    assert audit["test_prefs"]["sft_validation_overlap"] == 1
    assert audit["test_prefs"]["kept"] == 2
    chosen, rejected = examples["train_prefs"][0]
    assert chosen[0][0] == rejected[0][0] == config.bos_token_id
    for ids, labels, count in (chosen, rejected):
        assert labels[:len(ids) - count] == (-100,) * (len(ids) - count)
        assert labels[-count:] == ids[-count:]
        assert ids[-1] == labels[-1] == config.eos_token_id
    ids, labels, mask = dpo.collate(examples["train_prefs"][:2], torch.device("cpu"),
                                    config.eos_token_id)
    assert ids.shape == labels.shape == mask.shape
    assert torch.all(labels[~mask] == -100)
    assert torch.equal(ids[0, :len(chosen[0]) - chosen[2]],
                       ids[2, :len(rejected[0]) - rejected[2]])


def test_sequence_logps_and_dpo_gradient():
    logits = torch.tensor([[[0., 0., 0.], [2., 0., -1.], [0., 2., -1.],
                            [1., 1., 1.]]], requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2]])
    actual = dpo.sequence_logps(logits, labels)
    expected = (logits.log_softmax(-1)[0, 1, 1] +
                logits.log_softmax(-1)[0, 2, 2]).unsqueeze(0)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.all(logits.grad[0, 0] == 0)
    assert torch.all(logits.grad[0, 3] == 0)

    policy = torch.tensor([-2., -3., -4., -5.], requires_grad=True)
    reference = torch.tensor([-2., -3., -4., -5.])
    losses, margins, wins = dpo.dpo_objective(policy, reference, 0.1)
    torch.testing.assert_close(losses, torch.full((2,), torch.log(torch.tensor(2.))))
    torch.testing.assert_close(margins, torch.zeros(2))
    assert wins.tolist() == [True, True]
    losses.mean().backward()
    assert torch.all(policy.grad[:2] < 0)
    assert torch.all(policy.grad[2:] > 0)


def args_for(data, source, output=None, resume=None, stop=None):
    return SimpleNamespace(raw_data_dir=data, init_checkpoint=source,
                           output_dir=output, resume=resume, audit_only=False,
                           precision="fp32", max_length=32, batch_size=1,
                           grad_accum_steps=2, epochs=1, learning_rate=1e-3,
                           beta=0.1, min_lr_ratio=0.1, warmup_ratio=0,
                           weight_decay=0.1, beta1=0.9, beta2=0.95,
                           max_grad_norm=1., eval_every=1, save_every=1,
                           log_every=1, seed=2026, stop_after_steps=stop)


def test_paused_dpo_matches_uninterrupted_training(tiny_dpo, tmp_path):
    data, _, _, source = tiny_dpo
    ctx = ddp.Context(0, 1, torch.device("cpu"))
    paused = tmp_path / "paused"
    dpo.run(args_for(data, source, paused, stop=1), ctx)
    dpo.run(args_for(data, source, resume=paused / "checkpoint.pt"), ctx)
    full = tmp_path / "full"
    dpo.run(args_for(data, source, full), ctx)
    actual = torch.load(paused / "checkpoint.pt", weights_only=True)
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    assert actual["progress"] == expected["progress"]
    for key in actual["model_state_dict"]:
        torch.testing.assert_close(actual["model_state_dict"][key],
                                   expected["model_state_dict"][key], rtol=0, atol=0)
    loaded, _, _ = load_model(paused / "checkpoint.pt", torch.device("cpu"))
    assert loaded.num_parameters() > 0
    changed_source = torch.load(source, weights_only=True)
    changed_source["model_state_dict"]["norm.weight"][0] += 0.01
    torch.save(changed_source, source)
    with pytest.raises(ValueError, match="恢复约定"):
        dpo.run(args_for(data, source, resume=paused / "checkpoint.pt"), ctx)


def _two_rank_worker(rank, rendezvous, data, source, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        ctx = ddp.Context(rank, 2, torch.device("cpu"))
        args = args_for(Path(data), Path(source), Path(output))
        args.grad_accum_steps = 1
        dpo.run(args, ctx)
    finally:
        dist.destroy_process_group()


def test_two_rank_dpo_completes(tiny_dpo, tmp_path):
    data, _, _, source_path = tiny_dpo
    source = torch.load(source_path, weights_only=True)
    source["contract"]["world_size"] = 2
    source["rng_states"] = [source["rng_states"][0], source["rng_states"][0]]
    two_rank_source = tmp_path / "sft_two_rank.pt"
    torch.save(source, two_rank_source)
    output = tmp_path / "two_rank"
    mp.spawn(_two_rank_worker,
             args=(str(tmp_path / "rendezvous"), str(data), str(two_rank_source), str(output)),
             nprocs=2, join=True)
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["progress"]["step"] == 2
    assert saved["progress"]["validation"]["pairs"] == 2
    assert saved["contract"]["world_size"] == 2
