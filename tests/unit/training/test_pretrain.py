"""预训练回归检查，重点验证累积权重、epoch 尾部以及中断恢复的连续性。"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models

np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")
from dummym.models.llama_like import MiniLlamaConfig, MiniLlamaForCausalLM


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/train/pretrain.py"
spec = importlib.util.spec_from_file_location("pretrain_script", SCRIPT)
pretrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretrain)


def test_device_ordinal_is_relative_to_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert pretrain.resolve_device("cuda") == torch.device("cuda:0")
    assert pretrain.resolve_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(ValueError, match="仅可见 1 张 GPU") as error:
        pretrain.resolve_device("cuda:1")
    assert "CUDA_VISIBLE_DEVICES='1'" in str(error.value)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert pretrain.resolve_device("cuda:1") == torch.device("cuda:1")


def test_no_visible_gpu_has_clear_error_and_cpu_still_works(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="可见 GPU 数=0"):
        pretrain.resolve_device("cuda")
    assert pretrain.resolve_device("cpu") == torch.device("cpu")


@pytest.fixture
def prepared_data(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"<s>": 1, "</s>": 2, "<unk>": 0,
                                           **{f"word{i}": i + 3 for i in range(61)}}, unk_token="<unk>"))
    tokenizer.save(str(directory / "tokenizer.json"))
    config = MiniLlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=8, bos_token_id=1, eos_token_id=2,
                            attention_dropout=0.1)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"model_config": config.to_dict()}))
    stats = {}
    # 7 条训练序列不能整除有效 batch=4，3 条验证序列不能整除 micro-batch=2。
    # 这样的尾部能发现简单“固定除以累积次数”导致的错误，而不是只测整齐的 batch。
    for split, count in (("train", 7), ("validation", 3)):
        ids = np.arange(count * 8, dtype=np.uint16).reshape(count, 8) % 61 + 3
        ids[:, -1] = 2
        ids.astype("<u2").tofile(directory / f"{split}.bin")
        stats[split] = {"written_tokens": count * 8, "sequences": count, "bytes": count * 16}
    metadata = {"status": "complete", "format": {"dtype": "<u2", "sequence_length": 8},
                "tokenizer": {"file": "tokenizer.json", "sha256": pretrain.file_sha256(directory / "tokenizer.json"),
                              "vocab_size": 64, "eos_token_id": 2}, "splits": stats}
    (directory / "summary.json").write_text(json.dumps(metadata))
    return directory, model_path, config


def test_lr_warmup_and_decay_boundaries():
    factors = [pretrain.lr_factor(i, 6, 2, 0.1) for i in range(7)]
    assert factors[:3] == [0.5, 1.0, 1.0]
    assert factors[3] > factors[4] > factors[5]
    assert factors[5] == pytest.approx(0.1)
    assert factors[6] == factors[5]
    assert pretrain.lr_factor(0, 1, 0, 0.1) == 1


def test_accumulation_matches_single_batch_with_uneven_tail(prepared_data):
    directory, _, config = prepared_data
    config.attention_dropout = 0
    datasets, _, _ = pretrain.load_data(directory, config)
    torch.manual_seed(11)
    first = MiniLlamaForCausalLM(config)
    second = copy.deepcopy(first)
    args = SimpleNamespace(batch_size=2, precision="fp32", max_grad_norm=1.0,
                           learning_rate=0.001, beta1=0.9, beta2=0.95, weight_decay=0.1)
    indices = np.array([1, 3, 5])
    first_loss, _ = pretrain.train_update(first, pretrain.make_optimizer(first, args),
                                         datasets["train"], indices, args, torch.device("cpu"))
    args.batch_size = 3
    second_loss, _ = pretrain.train_update(second, pretrain.make_optimizer(second, args),
                                          datasets["train"], indices, args, torch.device("cpu"))
    assert first_loss == pytest.approx(second_loss, abs=1e-6)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)


def test_validation_weights_last_batch_and_restores_train_mode(prepared_data):
    directory, _, config = prepared_data
    datasets, _, _ = pretrain.load_data(directory, config)
    model = MiniLlamaForCausalLM(config).train()
    args = SimpleNamespace(batch_size=2, eval_batches=0, precision="fp32")
    loss, count = pretrain.evaluate(model, datasets["validation"], args, torch.device("cpu"))
    assert model.training
    assert count == 3
    model.eval()
    all_rows = datasets["validation"].batch(slice(None), torch.device("cpu"))
    with torch.no_grad():
        expected = model(all_rows, labels=all_rows).loss.item()
    assert loss == pytest.approx(expected, abs=1e-6)


def run_cli(monkeypatch, prepared_data, output, extra=()):
    directory, config, _ = prepared_data
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--data-dir", str(directory),
                        "--model-config", str(config), "--output-dir", str(output),
                        "--device", "cpu", "--precision", "fp32", "--batch-size", "2",
                        "--grad-accum-steps", "2", "--epochs", "2", "--warmup-steps", "1",
                        "--save-every", "2", "--eval-every", "2", "--cpu-threads", "1", *extra])
    pretrain.main()


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


def test_resume_matches_uninterrupted_training_across_epoch(monkeypatch, prepared_data, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run_cli(monkeypatch, prepared_data, full)
    run_cli(monkeypatch, prepared_data, resumed, ("--stop-after-steps", "1"))
    partial = torch.load(resumed / "checkpoint.pt", weights_only=True)
    assert partial["progress"]["step"] == 1
    assert partial["progress"]["next_sequence"] == 4
    assert partial["contract"]["total_steps"] == 4  # 暂停点不改变完整 LR 计划。
    run_cli(monkeypatch, prepared_data, resumed, ("--resume", str(resumed / "checkpoint.pt")))
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    for name in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "progress", "rng_state"):
        assert_identical(expected[name], actual[name])
    assert actual["progress"]["tokens_seen"] == 7 * 8 * 2
    assert actual["progress"]["epoch"] == 2
    assert actual["progress"]["next_sequence"] == 0
    assert json.loads((resumed / "summary.json").read_text())["status"] == "complete"

    # 相同路径的训练文件被替换时，恢复也必须拒绝；不只比较文件路径或大小。
    directory, _, _ = prepared_data
    data = np.fromfile(directory / "train.bin", dtype="<u2")
    data[0] = 6
    data.tofile(directory / "train.bin")
    with pytest.raises(ValueError, match="训练约定"):
        run_cli(monkeypatch, prepared_data, resumed, ("--resume", str(resumed / "checkpoint.pt")))


def test_rejects_mismatched_data_and_refuses_overwrite(monkeypatch, prepared_data, tmp_path):
    directory, _, config = prepared_data
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_cli(monkeypatch, prepared_data, output)
    with (directory / "train.bin").open("ab") as handle:
        handle.write(b"\x00\x00")
    with pytest.raises(ValueError, match="字节数"):
        pretrain.load_data(directory, config)
