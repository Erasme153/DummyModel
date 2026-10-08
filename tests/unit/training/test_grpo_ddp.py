"""GRPO reward isolation, group objective and checkpoint recovery."""

from __future__ import annotations

from datetime import timedelta
import math
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
from scripts.eval import grpo_baseline as baseline
from scripts.train import grpo_ddp as grpo
from scripts.train import pretrain as single
from scripts.train import pretrain_ddp as ddp


@pytest.fixture
def tiny_grpo(tmp_path):
    data = tmp_path / "gsm8k"
    data.mkdir()
    train = [{"question": f"Question {i}", "answer": f"work\n#### {i}"}
             for i in range(10)]
    train.append({"question": "Ultra question", "answer": "work\n#### 11"})
    pq.write_table(pa.Table.from_pylist(train), data / "train-00000.parquet")
    pq.write_table(pa.Table.from_pylist([{"question": "Official test", "answer": "SECRET"}]),
                   data / "test-00000.parquet")
    ultra = tmp_path / "ultra"
    ultra.mkdir()
    for split in ("train_sft", "test_prefs"):
        pq.write_table(pa.Table.from_pylist([{"prompt": "Ultra question"}]),
                       ultra / f"{split}-00000.parquet")
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
    _, audit = baseline.load_data(data, ultra, tokenizer, config, holdout_size=2,
                                  max_new_tokens=4, seed=2026, train_only=True)
    torch.manual_seed(7)
    model = MiniLlamaForCausalLM(config)
    source = tmp_path / "sft.pt"
    torch.save({"format_version": 3, "model_config": config.to_dict(),
                "model_state_dict": model.state_dict(),
                "contract": {"world_size": 1,
                             "data": {"dataset": "gsm8k",
                                      "tokenizer_sha256": single.file_sha256(tokenizer_path),
                                      "source_sha256": audit["source_sha256"],
                                      "test_questions_sha256": audit["test_questions_sha256"],
                                      "holdout_source_indices_sha256":
                                          audit["holdout_source_indices_sha256"]}},
                "tokenizer_path": str(tokenizer_path),
                "rng_states": [single.rng_state(torch.device("cpu"))]}, source)
    return data, ultra, tokenizer, config, source, audit


def args_for(data, ultra, source, output=None, resume=None, stop=None):
    return SimpleNamespace(data_dir=data, ultrafeedback_dir=ultra,
                           init_checkpoint=source, output_dir=output, resume=resume,
                           audit_only=False, precision="fp32", max_length=64,
                           holdout_size=2, max_new_tokens=4, group_size=2,
                           batch_size=1, grad_accum_steps=1, policy_epochs=2,
                           epochs=1, temperature=0.7, top_k=4, top_p=0.95,
                           learning_rate=1e-3, kl_coef=0.01, clip_eps=0.2,
                           min_lr_ratio=0.1, warmup_ratio=0, weight_decay=0.1,
                           beta1=0.9, beta2=0.95, max_grad_norm=1.0,
                           save_every=1, log_every=1, stop_after_steps=stop, seed=2026)


def test_group_advantages_and_clipped_objective():
    rewards = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.],
                            [1., 1., 1., 1.]])
    advantages = grpo.group_advantages(rewards)
    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.))
    assert advantages[0, 0] > 0 and (advantages[0, 1:] < 0).all()
    assert torch.count_nonzero(advantages[1:]) == 0

    labels = torch.tensor([[-100, -100, 2, 3], [-100, -100, 2, -100]])
    current = torch.tensor([[0., 0., 0.], [0., 0., 0.]], requires_grad=True)
    old = torch.zeros_like(current)
    reference = torch.zeros_like(current)
    loss, kl, clipped = grpo.grpo_objective(
        current, old, reference, labels, torch.tensor([1., -1.]),
        clip_eps=0.2, kl_coef=0.01)
    torch.testing.assert_close(loss, torch.tensor([-1., 1.]))
    torch.testing.assert_close(kl, torch.tensor(0.))
    torch.testing.assert_close(clipped, torch.tensor(0.))
    loss.sum().backward()
    assert current.grad[0, 0] == current.grad[1, 0] == current.grad[1, 2] == 0
    assert current.grad[0, 1] < 0 and current.grad[1, 1] > 0

    clipped_current = torch.tensor([[0., math.log(2), 0.]], requires_grad=True)
    clipped_loss, _, fraction = grpo.grpo_objective(
        clipped_current, torch.zeros_like(clipped_current),
        torch.zeros_like(clipped_current), labels[:1], torch.ones(1),
        clip_eps=0.2, kl_coef=0)
    torch.testing.assert_close(fraction, torch.tensor(0.5))
    clipped_loss.backward()
    assert clipped_current.grad[0, 1] == 0


def test_rollout_masks_prompt_and_keeps_baseline_split(tiny_grpo):
    data, ultra, tokenizer, config, _, audit = tiny_grpo
    datasets, actual = baseline.load_data(data, ultra, tokenizer, config,
                                          holdout_size=2, max_new_tokens=4,
                                          seed=2026, train_only=True)
    assert actual["holdout_source_indices_sha256"] == audit["holdout_source_indices_sha256"]
    assert len(datasets["train"]) == 8 and len(datasets["holdout"]) == 2
    assert "test" not in actual["source_sha256"]
    prompt = datasets["train"][0]["prompt_ids"]
    ids, labels, mask = grpo.collate_rollouts(
        [(prompt, (4, 2), 1), (prompt, (5,), 0)], torch.device("cpu"), config.eos_token_id)
    assert torch.all(labels[:, :len(prompt)] == -100)
    assert labels[0, len(prompt):len(prompt) + 2].tolist() == [4, 2]
    assert torch.all(labels[~mask] == -100)
    logits = torch.zeros(2, ids.shape[1], config.vocab_size, requires_grad=True)
    logps = grpo.token_logps(logits, labels)
    assert torch.all(logps[:, :len(prompt) - 1] == 0)
    logps.sum().backward()
    assert torch.all(logits.grad[:, :len(prompt) - 1] == 0)

    torch.manual_seed(7)
    model = MiniLlamaForCausalLM(config).eval()
    common = dict(group_size=2, max_new_tokens=4, temperature=0.7, top_k=4,
                  top_p=0.95, eos=config.eos_token_id, device=torch.device("cpu"),
                  precision="fp32")
    texts, lengths, completions = baseline.generate_group(
        model, prompt, tokenizer, generator=torch.Generator().manual_seed(7),
        return_token_ids=True, **common)
    old_texts, old_lengths = baseline.generate_group(
        model, prompt, tokenizer, generator=torch.Generator().manual_seed(7),
        **common)
    assert (texts, lengths) == (old_texts, old_lengths)
    assert [len(tokens) for tokens in completions] == lengths


def test_paused_grpo_matches_uninterrupted_training(tiny_grpo, tmp_path):
    data, ultra, _, _, source, _ = tiny_grpo
    ctx = ddp.Context(0, 1, torch.device("cpu"))
    paused = tmp_path / "paused"
    grpo.run(args_for(data, ultra, source, paused, stop=1), ctx)
    grpo.run(args_for(data, ultra, source, resume=paused / "checkpoint.pt", stop=2), ctx)
    full = tmp_path / "full"
    grpo.run(args_for(data, ultra, source, full, stop=2), ctx)
    actual = torch.load(paused / "checkpoint.pt", weights_only=True)
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    assert actual["progress"] == expected["progress"]
    for key in actual["model_state_dict"]:
        torch.testing.assert_close(actual["model_state_dict"][key],
                                   expected["model_state_dict"][key], rtol=0, atol=0)


def _two_rank_worker(rank, rendezvous, data, ultra, source, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        args = args_for(Path(data), Path(ultra), Path(source), Path(output), stop=1)
        grpo.run(args, ddp.Context(rank, 2, torch.device("cpu")))
    finally:
        dist.destroy_process_group()


def test_two_rank_grpo_completes(tiny_grpo, tmp_path):
    data, ultra, _, _, source_path, _ = tiny_grpo
    source = torch.load(source_path, weights_only=True)
    source["contract"]["world_size"] = 2
    source["rng_states"] *= 2
    two_rank_source = tmp_path / "sft_two_rank.pt"
    torch.save(source, two_rank_source)
    output = tmp_path / "two_rank"
    mp.spawn(_two_rank_worker,
             args=(str(tmp_path / "rendezvous"), str(data), str(ultra),
                   str(two_rank_source), str(output)), nprocs=2, join=True)
    saved = torch.load(output / "checkpoint.pt", weights_only=True)
    assert saved["progress"]["step"] == 1
    assert saved["progress"]["optimizer_updates"] == 2
    assert saved["contract"]["world_size"] == 2
