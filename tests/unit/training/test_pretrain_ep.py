"""Two-rank EP must match the replicated MoE's forward and global update."""

import copy
from datetime import timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
from tokenizers import Tokenizer, models
import yaml

from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM
from scripts.train import pretrain as single
from scripts.train import pretrain_ddp as ddp
from scripts.train import pretrain_ep as ep
from scripts.eval.base_eval import load_model


def ep_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        config = MiniLlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, max_position_embeddings=8,
                                 num_experts=4, experts_per_token=2,
                                 expert_intermediate_size=16)
        args = SimpleNamespace(learning_rate=1e-3, weight_decay=0.1, beta1=0.9,
                               beta2=0.95, max_grad_norm=1.0)
        for zero_token_experts in (False, True):
            torch.manual_seed(17)
            reference = MiniLlamaForCausalLM(config)
            if zero_token_experts:
                with torch.no_grad():
                    for layer in reference.layers:
                        layer.mlp.router.weight.zero_()
            model = copy.deepcopy(reference)
            for layer in model.layers:
                layer.mlp.shard_experts(rank, 2)
                assert set(layer.mlp.experts) == set(map(str, range(rank * 2, rank * 2 + 2)))
            shared, experts = ep.shared_and_expert_parameters(model)
            reference_optimizer = single.make_optimizer(reference, args)
            ep_optimizer = single.make_optimizer(model, args)
            generator = torch.Generator().manual_seed(120 + rank)
            tokens = torch.randint(3, 64, (2, 8), generator=generator)

            actual = model(tokens, labels=tokens)
            expected = reference(tokens, labels=tokens)
            torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual.loss, expected.loss, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(actual.router_stats["selected_counts"],
                                       expected.router_stats["selected_counts"])
            (expected.loss + 0.01 * expected.aux_loss).div(2).backward()
            for parameter in reference.parameters():
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            (actual.loss + 0.01 * actual.aux_loss).div(2).backward()
            ep.synchronize_shared_gradients(shared)
            reference_parameters = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad, reference_parameters[name].grad,
                                           atol=2e-5, rtol=2e-4)
            actual_norm = ep.clip_ep_grad_norm(shared, experts, 1.0, torch.device("cpu"))
            expected_norm = clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
            assert actual_norm == pytest.approx(float(expected_norm), rel=1e-5, abs=1e-6)
            reference_optimizer.step()
            ep_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, reference_parameters[name],
                                           atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="需要 Gloo")
def test_two_rank_ep_matches_replicated_forward_gradients_and_update(tmp_path):
    mp.spawn(ep_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def ep_resume_worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    root = Path(root)
    ctx = ddp.Context(rank, 2, torch.device("cpu"))
    try:
        base = ["pretrain_ep.py", "--data-dir", str(root / "data"),
                "--model-config", str(root / "moe.yaml"), "--device", "cpu",
                "--precision", "fp32", "--batch-size", "1", "--grad-accum-steps", "1",
                "--warmup-steps", "1", "--eval-every", "2", "--save-every", "2",
                "--log-every", "1", "--cpu-threads", "1"]
        for name, extra in (("full", []), ("resumed", ["--stop-after-steps", "1"]),
                            ("resumed", ["--resume", str(root / "resumed/checkpoint.pt")])):
            sys.argv = base + extra
            if not any(part == "--resume" for part in extra):
                sys.argv += ["--output-dir", str(root / name)]
            ep.run(ddp.parse_args(), ctx)
        dist.barrier()
        full_meta = torch.load(root / "full/checkpoint.pt", weights_only=True)
        resumed_meta = torch.load(root / "resumed/checkpoint.pt", weights_only=True)
        assert full_meta["progress"] == resumed_meta["progress"]
        assert full_meta["progress"]["step"] == 4
        assert full_meta["progress"]["tokens_seen"] == 7 * 8
        assert_same(full_meta["scheduler_state_dict"], resumed_meta["scheduler_state_dict"])
        assert full_meta["contract"]["parallelism"] == "ep2"
        for metadata, name in ((full_meta, "full"), (resumed_meta, "resumed")):
            assert len(metadata["shards"]) == 2
            assert all((root / name / part).is_file() for part in metadata["shards"])
        full = torch.load(root / "full" / full_meta["shards"][rank], weights_only=True)
        resumed = torch.load(root / "resumed" / resumed_meta["shards"][rank], weights_only=True)
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state"):
            assert_same(full[key], resumed[key])
        for name, value in full["model_state_dict"].items():
            if ".mlp.experts." not in name:
                other = value.clone()
                dist.broadcast(other, src=0)
                assert_same(value, other)
        expert_ids = {int(key.split(".mlp.experts.")[1].split(".")[0])
                      for key in full["model_state_dict"] if ".mlp.experts." in key}
        assert expert_ids == set(range(rank * 2, rank * 2 + 2))
    finally:
        dist.destroy_process_group()


@pytest.fixture
def ep_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"word{i}": i + 3 for i in range(61)}}
    Tokenizer(models.WordLevel(vocab, unk_token="<unk>")).save(str(data / "tokenizer.json"))
    config = MiniLlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=8,
                             bos_token_id=1, eos_token_id=2, attention_dropout=0.1,
                             num_experts=4, experts_per_token=2,
                             expert_intermediate_size=16)
    (tmp_path / "moe.yaml").write_text(yaml.safe_dump({"model_config": config.to_dict()}))
    splits = {}
    for split, count in (("train", 7), ("validation", 3)):
        rows = (np.arange(count * 8).reshape(count, 8) % 61 + 3).astype("<u2")
        rows[:, -1] = 2
        rows.tofile(data / f"{split}.bin")
        splits[split] = {"written_tokens": count * 8, "sequences": count, "bytes": count * 16}
    (data / "summary.json").write_text(json.dumps({
        "status": "complete", "format": {"dtype": "<u2", "sequence_length": 8},
        "splits": splits, "tokenizer": {"file": "tokenizer.json", "vocab_size": 64,
                                      "eos_token_id": 2,
                                      "sha256": single.file_sha256(data / "tokenizer.json")}}))
    return tmp_path


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="需要 Gloo")
def test_ep_training_resume_matches_full_run(ep_data):
    mp.spawn(ep_resume_worker, args=(str(ep_data / "rendezvous"), str(ep_data)),
             nprocs=2, join=True)
    model, config, checkpoint = load_model(ep_data / "full/checkpoint.pt", torch.device("cpu"))
    assert config.num_experts == 4
    assert all(f"layers.0.mlp.experts.{expert}.gate_proj.weight" in checkpoint["model_state_dict"]
               for expert in range(4))
    datasets, _, _ = single.load_data(ep_data / "data", config)
    args = SimpleNamespace(batch_size=1, precision="fp32", eval_batches=0)
    loss, count = single.evaluate(model, datasets["validation"], args, torch.device("cpu"))
    summary = json.loads((ep_data / "full/summary.json").read_text())
    assert count == 3
    assert loss == pytest.approx(summary["progress"]["validation_loss"], abs=1e-6)
