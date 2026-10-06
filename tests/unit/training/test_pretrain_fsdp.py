"""FSDP2 入口中可在无 GPU 环境验证的 checkpoint 与尾部验证约定。"""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.distributed.checkpoint as dcp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.train import pretrain_fsdp as trainer


def test_dcp_restores_lazy_adamw_state_and_next_update(tmp_path):
    torch.manual_seed(2026)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def update(current_model, current_optimizer, current_scheduler):
        current_optimizer.zero_grad()
        current_model(batch).square().sum().backward()
        current_optimizer.step()
        current_scheduler.step()

    update(model, optimizer, scheduler)
    checkpoint = tmp_path / "checkpoint"
    dcp.save(trainer.checkpoint_state_for_save(model, optimizer, step=1),
             checkpoint_id=str(checkpoint))

    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    trainer.load_training_state(checkpoint, restored, restored_optimizer, restored_scheduler,
                                {"progress": {"step": 1}, "scheduler_state_dict": scheduler.state_dict()})
    assert len(restored_optimizer.state) == len(optimizer.state) == 2
    update(model, optimizer, scheduler)
    update(restored, restored_optimizer, restored_scheduler)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_initial_checkpoint_does_not_advance_adamw(tmp_path):
    torch.manual_seed(2026)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "checkpoint"
    state = trainer.checkpoint_state_for_save(model, optimizer, step=0)
    assert set(state) == {"model"}
    assert len(optimizer.state) == 0
    dcp.save(state, checkpoint_id=str(checkpoint))

    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    trainer.load_training_state(checkpoint, restored, restored_optimizer, restored_scheduler,
                                {"progress": {"step": 0}, "scheduler_state_dict": scheduler.state_dict()})
    assert len(restored_optimizer.state) == 0
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    for current_model, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        current_model(batch).square().sum().backward()
        current_optimizer.step()
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(state["step"].item() == 1 for state in restored_optimizer.state.values())


def test_checkpoint_contract_and_previous_fallback(tmp_path):
    previous = tmp_path / "checkpoint.prev"
    previous.mkdir()
    contract = {"parallelism": "fsdp2", "shard_policy": "blocks", "world_size": 2}
    torch.save({"format_version": 3, "contract": contract, "rng_states": [{}, {}]},
               previous / "trainer.pt")
    ctx = SimpleNamespace(world_size=2)
    metadata, path = trainer.load_metadata(tmp_path / "checkpoint", contract, ctx)
    assert path == previous
    assert metadata["format_version"] == 3
    with pytest.raises(ValueError, match="训练约定一致"):
        trainer.load_metadata(tmp_path / "checkpoint", {**contract, "shard_policy": "root"}, ctx)
    with pytest.raises(ValueError, match="FSDP2 checkpoint 目录"):
        trainer.load_metadata(tmp_path / "ddp_checkpoint.pt", contract, ctx)


def test_validation_uses_dummy_to_match_forward_counts(monkeypatch):
    class Dataset:
        def __len__(self):
            return 5

        def batch(self, selection, device):
            return torch.ones(len(range(*selection.indices(5))), 4)

    class Model:
        training = True

        def __init__(self):
            self.calls = 0

        def eval(self):
            self.training = False

        def train(self, state=True):
            self.training = state

        def __call__(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(loss=torch.tensor(2.0))

    def finish_reduce(tensor, op=None):
        tensor[0] = 10.0
        tensor[1] = 5.0

    monkeypatch.setattr(trainer.dist, "all_reduce", finish_reduce)
    model = Model()
    args = SimpleNamespace(batch_size=1, eval_batches=0, precision="fp32")
    ctx = SimpleNamespace(world_size=2, rank=0, device=torch.device("cpu"))
    loss, count = trainer.evaluate(model, Dataset(), args, ctx)
    assert (loss, count, model.calls, model.training) == (2.0, 5, 3, True)
