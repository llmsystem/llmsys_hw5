import copy
import json
import random
from pathlib import Path
import pytest
import torch
from torch import nn
from assignment import (
    data_parallel as dp,
    pipeline as pp,
    finetune as ft,
    inference as inf,
)
from assignment.runtime import LoRALinear
from grading.fixtures import model


def test_partitions():
    state = random.getstate()
    for size, world in [(0, 3), (2, 5), (17, 3), (32, 2), (101, 7)]:
        result = dp.partition_indices(size, world, 19)
        assert len(result) == world
        assert sorted(i for part in result for i in part) == list(range(size))
        assert [len(p) for p in result] == [
            size // world + (r < size % world) for r in range(world)
        ]
        assert result == dp.partition_indices(size, world, 19)
    assert random.getstate() == state
    assert dp.partition_indices(100, 3, 19) != dp.partition_indices(100, 3, 20)
    for size, world in [(-1, 2), (3, 0), (3, -1)]:
        with pytest.raises(ValueError):
            dp.partition_indices(size, world)


def test_schedule():
    for m, n in [(0, 2), (1, 1), (1, 4), (7, 3), (6, 2), (3, 3)]:
        waves = list(pp.clock_cycles(m, n))
        assert len(waves) == (m + n - 1 if m else 0)
        all_pairs = []
        for tick, wave in enumerate(waves):
            assert wave and wave == sorted(wave, key=lambda pair: pair[1])
            assert len({s for _, s in wave}) == len(wave)
            for i, j in wave:
                assert 0 <= i < m and 0 <= j < n and i + j == tick
            all_pairs.extend(wave)
        assert sorted(all_pairs) == [(i, j) for i in range(m) for j in range(n)]
    for m, n in [(-1, 1), (3, 0)]:
        with pytest.raises(ValueError):
            list(pp.clock_cycles(m, n))


def reference_schedule(m, n):
    # Fixture: isolates forward/backward tests from scheduling implementation.
    return [[(t - j, j) for j in range(n) if 0 <= t - j < m] for t in range(m + n - 1)]


def devices():
    import os

    return (
        ["cuda:0", "cuda:1"]
        if os.environ.get("HW56_ACCELERATOR") == "cuda"
        else ["cpu", "cpu"]
    )


def build_pipe(split, count=2):
    torch.manual_seed(23)
    first = nn.Sequential(nn.Linear(4, 7), nn.Tanh())
    last = nn.Linear(7, 3)
    stages = (
        [first, last]
        if count == 2
        else [first, nn.Linear(7, 7), last]
        if count == 3
        else [first]
    )
    reference = copy.deepcopy(nn.Sequential(*stages))
    ds = [devices()[i % 2] for i in range(count)]
    for stage, device in zip(stages, ds):
        stage.to(device)
    reference.to(ds[0])
    return pp.Pipe(stages, ds, split), reference


@pytest.mark.parametrize("count", [1, 2, 3])
def test_pipeline_forward(monkeypatch, count):
    monkeypatch.setattr(pp, "clock_cycles", reference_schedule)
    for size, split in [(1, 1), (7, 3), (3, 8), (8, 1)]:
        pipe, ref = build_pipe(split, count)
        try:
            x = torch.randn(size, 4, device=devices()[0])
            for _ in range(3):
                torch.testing.assert_close(
                    pipe(x).to(devices()[0]), ref(x), rtol=1e-5, atol=1e-6
                )
            assert pipe(x).device == pipe.devices[-1]
            assert len(list(pipe.parameters())) == len(list(ref.parameters()))
            expected = []
            for wave in reference_schedule((size + split - 1) // split, count):
                expected += [("submit", s) for _, s in wave] + [
                    ("receive", s) for _, s in wave
                ]
            assert pipe.workers.events[: len(expected)] == expected, (
                "Enqueue entire wave before waiting"
            )
            with torch.no_grad():
                assert not pipe(x).requires_grad
        finally:
            pipe.close()
        assert not any(t.is_alive() for t in pipe.workers.threads)


@pytest.mark.parametrize("count", [2, 3])
def test_pipeline_backward(monkeypatch, count):
    monkeypatch.setattr(pp, "clock_cycles", reference_schedule)
    pipe, ref = build_pipe(3, count)
    optimizer = torch.optim.SGD(pipe.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(ref.parameters(), lr=0.01)
    try:
        for _ in range(3):
            x = torch.randn(7, 4, device=devices()[0], requires_grad=True)
            y = x.detach().clone().requires_grad_()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            pipe(x).square().mean().backward()
            ref(y).square().mean().backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
            for p, q in zip(pipe.parameters(), ref.parameters()):
                assert p.grad is not None
                torch.testing.assert_close(
                    p.grad.to(q.device), q.grad, rtol=1e-4, atol=1e-6
                )
            optimizer.step()
            reference_optimizer.step()
            for p, q in zip(pipe.parameters(), ref.parameters()):
                torch.testing.assert_close(p.to(q.device), q, rtol=1e-4, atol=1e-6)
    finally:
        pipe.close()


def test_pipeline_error(monkeypatch):
    monkeypatch.setattr(pp, "clock_cycles", reference_schedule)

    class Broken(nn.Module):
        def forward(self, x):
            raise RuntimeError("sentinel worker failure")

    with pp.Pipe([Broken()], ["cpu"], 2) as pipe:
        with pytest.raises(RuntimeError, match="sentinel worker failure"):
            pipe(torch.ones(3, 2))
    assert not any(t.is_alive() for t in pipe.workers.threads)


def test_config():
    for world, micro, acc in [(2, 2, 2), (1, 3, 4), (4, 1, 3)]:
        for dtype in ["fp32", "fp16"]:
            c = ft.build_config(world, micro, acc, dtype)
            assert c["train_batch_size"] == world * micro * acc
            assert c["train_micro_batch_size_per_gpu"] == micro
            assert c["gradient_accumulation_steps"] == acc
            assert c["zero_optimization"]["stage"] == 2
            assert c.get("gradient_clipping", 0) == 0
            assert c.get("fp16", {}).get("enabled", False) == (dtype == "fp16")
            assert not c.get("bf16", {}).get("enabled", False)
            assert c.get("zero_allow_untested_optimizer") is True
            assert all(
                c["zero_optimization"].get(k, {}).get("device", "none") == "none"
                for k in ["offload_param", "offload_optimizer"]
            )
    for args in [
        (0, 1, 1, "fp32"),
        (2, 0, 1, "fp32"),
        (2, 1, 0, "fp32"),
        (2, 1, 1, "bf16"),
    ]:
        with pytest.raises(ValueError):
            ft.build_config(*args)


def test_lora_targets():
    torch.manual_seed(5)
    net = nn.Sequential(model(), nn.Linear(2, 1)).double().to(devices()[0])
    original = copy.deepcopy(net)
    x = torch.randn(3, 4, dtype=torch.float64, device=devices()[0])
    result = ft.apply_lora(net, ["0.0", "0.2"], rank=3, alpha=6)
    assert result is net
    for name in ["0.0", "0.2"]:
        layer = net.get_submodule(name)
        assert (
            isinstance(layer, LoRALinear)
            and layer.lora_A.shape[0] == 3
            and layer.scale == 2
        )
    torch.testing.assert_close(net(x), original(x))
    assert {name for name, p in net.named_parameters() if p.requires_grad} == {
        "0.0.lora_A",
        "0.0.lora_B",
        "0.2.lora_A",
        "0.2.lora_B",
    }
    assert all(p.dtype == torch.float64 for p in net.parameters())
    for names in [[], ["0.0", "bad"], ["0.0", "0.0"], ["0.1"]]:
        net = copy.deepcopy(original)
        state = {n: p.requires_grad for n, p in net.named_parameters()}
        with pytest.raises(ValueError):
            ft.apply_lora(net, names)
        assert not any(isinstance(m, LoRALinear) for m in net.modules())
        assert state == {n: p.requires_grad for n, p in net.named_parameters()}


def adapted_reference():
    torch.manual_seed(77)
    net = model().to(devices()[0])
    net.requires_grad_(False)
    net[0] = LoRALinear(net[0], 2, 4)
    net[2] = LoRALinear(net[2], 2, 4)
    return net


def test_adapter_persistence(tmp_path):
    net = adapted_reference()
    fresh = copy.deepcopy(net)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "lora_" in name:
                p.add_(0.1)
    path = tmp_path / "adapter.pt"
    ft.save_adapter(net, path)
    state = torch.load(path, weights_only=True)
    assert all(value.device.type == "cpu" for value in state.values())
    assert set(state) == {n for n, p in net.named_parameters() if "lora_" in n}
    base = {n: p.clone() for n, p in fresh.named_parameters() if "lora_" not in n}
    ft.load_adapter(fresh, path)
    x = torch.randn(3, 4, device=devices()[0])
    torch.testing.assert_close(net(x), fresh(x))
    for name, p in fresh.named_parameters():
        if name in base:
            assert torch.equal(p, base[name])
    for bad in [
        {},
        dict(state, extra=torch.zeros(1)),
        {k: torch.zeros(1) for k in state},
    ]:
        before = {n: p.clone() for n, p in fresh.named_parameters()}
        torch.save(bad, path)
        with pytest.raises(ValueError):
            ft.load_adapter(fresh, path)
        assert all(torch.equal(p, before[n]) for n, p in fresh.named_parameters())


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.syncs = 0

    def generate(self, prompts, params):
        self.calls.append((prompts, params))
        assert params["temperature"] == 0
        return [{"text": p + " ☃", "output_ids": [len(p), 3]} for p in prompts]

    def synchronize(self):
        self.syncs += 1

    def shutdown(self):
        self.closed = True


def requests():
    return [
        {"id": str(i), "prompt": p}
        for i, p in enumerate(["red", "blue", "red", "green", "你好"])
    ]


def test_inference_batching():
    for size in [1, 2, 10]:
        engine = FakeEngine()
        rs = requests()
        records = inf.generate_records(engine, rs, size, 7)
        assert [r["id"] for r in records] == [r["id"] for r in rs]
        assert [r["prompt"] for r in records] == [r["prompt"] for r in rs]
        assert [r["output"] for r in records] == [r["prompt"] + " ☃" for r in rs]
        assert [r["output_ids"] for r in records] == [[len(r["prompt"]), 3] for r in rs]
        assert len(engine.calls) == (len(rs) + size - 1) // size
        assert all(c[1]["max_new_tokens"] == 7 for c in engine.calls)
        assert not engine.closed
    assert inf.generate_records(FakeEngine(), [], 3) == []
    for size, tokens in [(0, 8), (1, 0)]:
        with pytest.raises(ValueError):
            inf.generate_records(FakeEngine(), requests(), size, tokens)
    with pytest.raises(ValueError):
        inf.generate_records(FakeEngine(), [requests()[0]] * 2, 2)

    class Missing(FakeEngine):
        def generate(self, *args):
            return []

    with pytest.raises(ValueError):
        inf.generate_records(Missing(), requests(), 2)


def test_inference_jsonl(tmp_path):
    rows = [
        {"id": str(i), "prompt": "a\nb", "output": "你好", "output_ids": [1, 2]}
        for i in range(23)
    ]
    path = tmp_path / "outputs.jsonl"
    inf.write_jsonl(rows, path)
    assert [
        json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()
    ] == rows


def test_inference_benchmark(monkeypatch):
    # Oracle batching avoids deducting batching bugs again in the timing criterion.
    def records(engine, rs, bs, max_new_tokens):
        outputs = engine.generate(
            [r["prompt"] for r in rs],
            {"temperature": 0, "max_new_tokens": max_new_tokens},
        )
        return [{"output_ids": o["output_ids"]} for o in outputs]

    monkeypatch.setattr(inf, "generate_records", records)
    ticks = iter([1.0, 3.0, 10.0, 13.0, 20.0, 25.0])
    monkeypatch.setattr(inf.time, "perf_counter", lambda: next(ticks))
    e = FakeEngine()
    stats = inf.benchmark(e, requests(), 2, repeats=3)
    assert (
        stats["output_tokens"] == 30
        and stats["seconds"] == 10
        and stats["tokens_per_second"] == 3
    )
    assert len(e.calls) == 4 and e.syncs == 6 and e.closed

    class Broken(FakeEngine):
        def generate(self, *a):
            raise RuntimeError("generation failed")

    e = Broken()
    with pytest.raises(RuntimeError):
        inf.benchmark(e, requests(), 2)
    assert e.closed
