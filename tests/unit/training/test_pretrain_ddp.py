"""真实双进程/Gloo 测试，不需要 GPU；覆盖同步、非均匀尾部和逐 rank RNG 恢复。"""

from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer, models

np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.train import pretrain_ddp as trainer
from scripts.train.pretrain_hyperball import HyperballWithAuxAdamW


class ToyDataset:
    def __init__(self, count):
        self.rows = torch.arange(count * 8).reshape(count, 8) % 61 + 3

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, device):
        return self.rows[indices].to(device)


def small_config(dropout=0.0):
    return trainer.MiniLlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8,
        bos_token_id=1, eos_token_id=2, attention_dropout=dropout)


def init_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    return trainer.Context(rank, 2, torch.device("cpu"))


def update_worker(rank, rendezvous):
    ctx = init_worker(rank, rendezvous)
    try:
        args = SimpleNamespace(batch_size=2, precision="fp32", max_grad_norm=1.0,
                               learning_rate=0.001, beta1=0.9, beta2=0.95,
                               weight_decay=0.1, eval_batches=0)
        dataset = ToyDataset(8)
        # 8: 两个完整累积 micro-batch；7: 不等长尾部；5: rank 1 最后一轮为空；
        # 1: rank 1 整个更新都没有真实样本。后三者不能用固定 loss/accum 加权。
        for count in (8, 7, 5, 1):
            torch.manual_seed(2026)
            raw = trainer.MiniLlamaForCausalLM(small_config())
            reference = copy.deepcopy(raw)
            model = DDP(raw, broadcast_buffers=False)
            actual_loss, actual_norm = trainer.train_update(
                model, trainer.single.make_optimizer(raw, args), dataset,
                np.arange(count), args, ctx)
            reference_args = copy.copy(args)
            reference_args.batch_size = count
            expected_loss, expected_norm = trainer.single.train_update(
                reference, trainer.single.make_optimizer(reference, args), dataset,
                np.arange(count), reference_args, ctx.device)
            assert actual_loss == pytest.approx(expected_loss, abs=1e-6)
            assert actual_norm == pytest.approx(expected_norm, rel=1e-5, abs=1e-6)
            for actual, expected in zip(raw.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
                # 同一次 DDP 更新后，两 rank 的参数必须逐元素相同。
                other = actual.detach().clone()
                dist.broadcast(other, src=0)
                torch.testing.assert_close(actual, other, atol=0, rtol=0)

        validation = ToyDataset(3)
        for batch_size, limit in ((2, 0), (1, 1)):
            args.batch_size, args.eval_batches = batch_size, limit
            actual_loss, actual_count = trainer.evaluate(model, validation, args, ctx)
            expected_loss, expected_count = trainer.single.evaluate(raw, validation, args, ctx.device)
            assert actual_count == expected_count
            assert actual_loss == pytest.approx(expected_loss, abs=1e-6)
            assert model.training

        # rank 0 写盘异常不能让另一个进程永久等在 collective 上。
        def fail_write():
            raise OSError("test write failure")
        with pytest.raises(RuntimeError, match="test write failure"):
            trainer.rank_zero_call(ctx, fail_write)
    finally:
        dist.destroy_process_group()


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


def resume_worker(rank, rendezvous, directory, optimizer_name):
    ctx = init_worker(rank, rendezvous)
    root = Path(directory)
    output_root = root / optimizer_name
    try:
        # 含 dropout，确保测试确实依赖恢复 RNG，而不是仅加载模型/优化器。
        base = ["pretrain_ddp.py", "--data-dir", str(root / "data"),
                "--model-config", str(root / "model.yaml"), "--device", "cpu",
                "--precision", "fp32", "--batch-size", "2", "--grad-accum-steps", "1",
                "--epochs", "2", "--train-sequences", "5", "--warmup-steps", "1",
                "--eval-every", "2", "--save-every", "2"]
        if optimizer_name == "muon":
            base += ["--optimizer", "muon", "--muon-lr", "0.02", "--adamw-lr", "0.001",
                     "--record-update-norms"]
        elif optimizer_name in ("muonh", "adamh"):
            base += ["--optimizer", optimizer_name, "--hyperball-lr", "0.005",
                     "--adamw-lr", "0.001", "--record-update-norms"]
        for name, extra in (("full", []), ("resumed", ["--stop-after-steps", "1"]),
                            ("resumed", ["--resume", str(output_root / "resumed/checkpoint.pt")])):
            sys.argv = base + ["--output-dir", str(output_root / name)] + extra
            trainer.run(trainer.parse_args(), ctx)
        dist.barrier()
        full = torch.load(output_root / "full/checkpoint.pt", weights_only=True)
        resumed = torch.load(output_root / "resumed/checkpoint.pt", weights_only=True)
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                    "progress", "rng_states"):
            assert_identical(full[key], resumed[key])
        assert resumed["progress"]["tokens_seen"] == 5 * 8 * 2
        assert resumed["progress"]["step"] == 4
        assert resumed["progress"]["next_sequence"] == 0
        assert resumed["contract"]["recipe"]["train_sequences"] == 5
        if optimizer_name == "muon":
            assert resumed["contract"]["recipe"]["optimizer"] == "muon"
            assert resumed["contract"]["recipe"]["muon_lr"] == 0.02
            assert resumed["contract"]["recipe"]["adamw_lr"] == 0.001
        elif optimizer_name in ("muonh", "adamh"):
            assert resumed["contract"]["recipe"]["optimizer"] == optimizer_name
            assert resumed["contract"]["recipe"]["hyperball_lr"] == 0.005
            assert resumed["contract"]["recipe"]["adamw_lr"] == 0.001
        # 推理脚本使用未包 DDP 的模型；不能让 checkpoint 权重带 module. 前缀。
        loaded = trainer.MiniLlamaForCausalLM(small_config(0.1))
        loaded.load_state_dict(resumed["model_state_dict"], strict=True)
        wrong = copy.deepcopy(resumed["contract"])
        wrong["world_size"] = 1
        with pytest.raises(ValueError, match="训练约定"):
            trainer.load_resume(output_root / "resumed/checkpoint.pt", wrong, ctx)
        wrong = copy.deepcopy(resumed["contract"])
        wrong["recipe"]["train_sequences"] = 6
        with pytest.raises(ValueError, match="训练约定"):
            trainer.load_resume(output_root / "resumed/checkpoint.pt", wrong, ctx)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="需要 Gloo")
def test_two_ranks_match_single_update_and_validation(tmp_path):
    mp.spawn(update_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.fixture
def prepared_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{f"word{i}": i + 3 for i in range(61)}}
    Tokenizer(models.WordLevel(vocab, unk_token="<unk>")).save(str(data / "tokenizer.json"))
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({"model_config": small_config(0.1).to_dict()}))
    splits = {}
    for split, count in (("train", 7), ("validation", 3)):
        rows = ToyDataset(count).rows.numpy().astype("<u2")
        rows[:, -1] = 2
        rows.tofile(data / f"{split}.bin")
        splits[split] = {"written_tokens": count * 8, "sequences": count, "bytes": count * 16}
    (data / "summary.json").write_text(json.dumps({
        "status": "complete", "format": {"dtype": "<u2", "sequence_length": 8}, "splits": splits,
        "tokenizer": {"file": "tokenizer.json", "vocab_size": 64, "eos_token_id": 2,
                      "sha256": trainer.single.file_sha256(data / "tokenizer.json")}}))
    return tmp_path


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="需要 Gloo")
@pytest.mark.parametrize("optimizer_name", ["adamw", "muon", "muonh", "adamh"])
def test_two_rank_resume_restores_rng_and_epoch_tail(prepared_data, optimizer_name):
    mp.spawn(resume_worker,
             args=(str(prepared_data / "rendezvous"), str(prepared_data), optimizer_name),
             nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="需要 Gloo")
def test_python_and_torchrun_entrypoints(prepared_data):
    # 真实 torchrun 验证 env:// 初始化及参数解析；不依赖测试进程手工建好的进程组。
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    root = prepared_data
    for workers, accum in ((1, 2), (2, 1)):
        launch = [sys.executable]
        if workers > 1:
            launch += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
        output = root / f"cli_{workers}"
        command = launch + [str(ROOT / "scripts/train/pretrain_ddp.py"),
                           "--data-dir", str(root / "data"), "--model-config", str(root / "model.yaml"),
                           "--output-dir", str(output), "--device", "cpu", "--precision", "fp32",
                           "--batch-size", "2", "--grad-accum-steps", str(accum),
                           "--warmup-steps", "0", "--cpu-threads", "1", "--stop-after-steps", "1"]
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"world_size={workers} global_batch=4" in result.stdout
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "paused"
        assert summary["progress"]["tokens_seen"] == 32
        assert len(list((output / "tensorboard").glob("events.out.tfevents.*"))) == 1


def test_rejects_old_checkpoint_format(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format_version": 1, "contract": {}}, path)
    with pytest.raises(ValueError, match="v2 checkpoint"):
        trainer.load_resume(path, {}, trainer.Context(0, 1, torch.device("cpu")))


def test_muon_groups_tied_embedding_and_records_actual_update():
    torch.manual_seed(2026)
    model = trainer.MiniLlamaForCausalLM(small_config())
    args = SimpleNamespace(muon_lr=0.02, adamw_lr=0.001, learning_rate=0.001,
                           muon_momentum=0.95, muon_ns_steps=5, weight_decay=0.1,
                           beta1=0.9, beta2=0.95, batch_size=2, precision="fp32",
                           max_grad_norm=1.0)
    optimizer = trainer.MuonWithAuxAdamW(model, args)
    groups = optimizer.param_groups
    assert [group["lr"] for group in groups] == [0.02, 0.001, 0.001]
    assert model.embed_tokens.weight is model.lm_head.weight
    assert any(model.embed_tokens.weight is p for p in groups[1]["params"])
    assert all(model.embed_tokens.weight is not p for p in groups[0]["params"])
    assert {id(p) for group in groups for p in group["params"]} == {id(p) for p in model.parameters()}
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))

    original = [p.detach().clone() for p in model.parameters()]
    metrics = {}
    loss, norm = trainer.train_update(model, optimizer, ToyDataset(4), np.arange(4), args,
                                      trainer.Context(0, 1, torch.device("cpu")), metrics)
    assert loss > 0 and norm > 0
    expected_parameter = sum(p.detach().square().sum().item() for p in model.parameters()) ** 0.5
    expected_update = sum((p.detach() - old).square().sum().item()
                          for p, old in zip(model.parameters(), original)) ** 0.5
    assert metrics["parameter_norm"] == pytest.approx(expected_parameter, rel=1e-6)
    assert metrics["update_norm"] == pytest.approx(expected_update, rel=1e-6)
    assert metrics["update_to_parameter_ratio"] == pytest.approx(expected_update / expected_parameter)
    assert metrics["muon_update_norm"] > 0
    assert metrics["adamw_update_norm"] > 0


def test_muon_auxiliary_update_matches_adamw():
    model = trainer.MiniLlamaForCausalLM(small_config())
    args = SimpleNamespace(muon_lr=0.02, adamw_lr=0.001, learning_rate=0.001,
                           muon_momentum=0.95, muon_ns_steps=5, weight_decay=0.1,
                           beta1=0.9, beta2=0.95)
    optimizer = trainer.MuonWithAuxAdamW(model, args)
    embedding = model.embed_tokens.weight
    reference = torch.nn.Parameter(embedding.detach().clone())
    reference_optimizer = torch.optim.AdamW([reference], lr=0.001, weight_decay=0.1,
                                             betas=(0.9, 0.95), eps=1e-8)
    for seed in (1, 2):
        torch.manual_seed(seed)
        gradient = torch.randn_like(embedding)
        embedding.grad = gradient.clone()
        reference.grad = gradient.clone()
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(embedding, reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("optimizer_name", ["muonh", "adamh"])
def test_hyperball_keeps_matrix_radii_and_auxiliary_adamw(optimizer_name):
    torch.manual_seed(2026)
    model = trainer.MiniLlamaForCausalLM(small_config())
    args = SimpleNamespace(optimizer=optimizer_name, hyperball_lr=0.01,
                           adamw_lr=0.001, learning_rate=0.001,
                           muon_momentum=0.95 if optimizer_name == "muonh" else None,
                           muon_ns_steps=5 if optimizer_name == "muonh" else None,
                           weight_decay=0.1, beta1=0.9, beta2=0.95,
                           batch_size=2, precision="fp32", max_grad_norm=1.0)
    optimizer = HyperballWithAuxAdamW(model, args)
    hidden = optimizer.param_groups[0]["params"]
    embedding = model.embed_tokens.weight
    reference = torch.nn.Parameter(embedding.detach().clone())
    reference_optimizer = torch.optim.AdamW([reference], lr=0.001, weight_decay=0.1,
                                             betas=(0.9, 0.95), eps=1e-8)
    assert model.embed_tokens.weight is model.lm_head.weight
    assert all(embedding is not p for p in hidden)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()}
    initial = {p: p.detach().clone() for p in hidden}
    radii = {p: optimizer.state[p]["hyperball_radius"] for p in hidden}
    for seed in (1, 2):
        torch.manual_seed(seed)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        reference.grad = embedding.grad.clone()
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(embedding, reference, rtol=1e-6, atol=1e-7)
        for p in hidden:
            assert p.detach().norm().item() == pytest.approx(radii[p], rel=1e-6)
    assert any(not torch.equal(p.detach(), initial[p]) for p in hidden)
    assert all("hyperball_radius" in optimizer.state[p] for p in hidden)


def test_hyperball_projection_uses_relative_step_and_fixed_radius():
    weight = torch.nn.Parameter(torch.tensor([[3.0, 0.0], [0.0, 4.0]]))
    direction = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    HyperballWithAuxAdamW._projected_step(weight, direction, radius=5.0, learning_rate=0.1)
    expected = torch.tensor([[3.0, -0.5], [0.0, 4.0]])
    expected *= 5.0 / expected.norm()
    torch.testing.assert_close(weight, expected)
    assert weight.detach().norm().item() == pytest.approx(5.0)


@pytest.mark.parametrize("optimizer_name", ["muonh", "adamh"])
def test_hyperball_script_defaults_and_tensorboard_metrics(prepared_data, optimizer_name):
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    output = prepared_data / f"{optimizer_name}_cli"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
               str(ROOT / "scripts/train/pretrain_hyperball.py"),
               "--data-dir", str(prepared_data / "data"),
               "--model-config", str(prepared_data / "model.yaml"),
               "--output-dir", str(output), "--device", "cpu", "--precision", "fp32",
               "--batch-size", "2", "--grad-accum-steps", "1", "--warmup-steps", "0",
               "--cpu-threads", "1", "--hyperball-lr", "0.005", "--stop-after-steps", "1"]
    if optimizer_name == "adamh":
        command += ["--optimizer", "adamh"]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["contract"]["recipe"]["optimizer"] == optimizer_name
    assert summary["contract"]["recipe"]["hyperball_lr"] == 0.005
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert any("hyperball_radius" in state
               for state in checkpoint["optimizer_state_dict"]["state"].values())
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(output / "tensorboard"))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    assert {f"train/{optimizer_name}_learning_rate", "train/adamw_learning_rate",
            f"train/{optimizer_name}_parameter_norm", f"train/{optimizer_name}_update_norm",
            f"train/{optimizer_name}_update_to_parameter_ratio"} <= tags


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="需要 Gloo")
def test_muon_script_defaults_and_tensorboard_metrics(prepared_data):
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    output = prepared_data / "muon_cli"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
               str(ROOT / "scripts/train/pretrain_muon.py"),
               "--data-dir", str(prepared_data / "data"),
               "--model-config", str(prepared_data / "model.yaml"),
               "--output-dir", str(output), "--device", "cpu", "--precision", "fp32",
               "--batch-size", "2", "--grad-accum-steps", "1", "--warmup-steps", "0",
               "--cpu-threads", "1", "--stop-after-steps", "1"]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["contract"]["recipe"]["optimizer"] == "muon"
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(output / "tensorboard"))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    assert {"train/muon_learning_rate", "train/adamw_learning_rate", "train/parameter_norm",
            "train/update_norm", "train/update_to_parameter_ratio", "train/muon_update_norm",
            "train/adamw_update_norm"} <= tags
    assert events.Scalars("train/muon_learning_rate")[0].value == pytest.approx(0.02)
    assert events.Scalars("train/adamw_learning_rate")[0].value == pytest.approx(0.001)
