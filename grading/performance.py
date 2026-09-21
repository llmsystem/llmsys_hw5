"""Two-V100 training benchmarks. Timings always include forward/backward/update.

Run through grade.py so correctness prerequisites and infrastructure checks apply.
"""

import argparse
import copy
import datetime
import json
import math
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from assignment import data_parallel
from assignment.pipeline import Pipe

WIDTH = 2048
DEPTH = 8
ROWS = 8192
SPLIT = 1024
WARMUP = 2
STEPS = 3
REPEATS = 5
THRESHOLDS = {"dp": 1.5, "pipeline": 1.1}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        return x + 0.1 * torch.nn.functional.gelu(self.linear(x))


def model():
    return nn.Sequential(*(Block() for _ in range(DEPTH)))


def sync(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def serial_step(net, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(net(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def timed(step, devices, steps=STEPS):
    sync(devices)
    start = time.perf_counter()
    for _ in range(steps):
        loss = step()
        assert isinstance(loss, float) and math.isfinite(loss)
    sync(devices)
    return (time.perf_counter() - start) / steps


def write_result(case, baseline, parallel, output):
    ratios = [a / b for a, b in zip(baseline, parallel)]
    speedup = statistics.median(ratios)
    result = {
        "case": case,
        "workload": {
            "width": WIDTH,
            "depth": DEPTH,
            "global_rows": ROWS,
            "pipeline_microbatch_rows": SPLIT,
            "dtype": "float32",
            "warmup_steps": WARMUP,
            "steps_per_repeat": STEPS,
            "repeats": REPEATS,
        },
        "baseline_seconds_per_step": baseline,
        "parallel_seconds_per_step": parallel,
        "paired_speedups": ratios,
        "median_speedup": speedup,
        "baseline_rows_per_second": ROWS / statistics.median(baseline),
        "parallel_rows_per_second": ROWS / statistics.median(parallel),
        "threshold": THRESHOLDS[case],
        "passed": speedup >= THRESHOLDS[case],
        "parameter_updates_verified": True,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert result["passed"], (
        f"{case} median speedup {speedup:.3f}x < {THRESHOLDS[case]:.2f}x"
    )


def dp_worker(rank, port, output):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=180),
    )
    try:
        torch.manual_seed(531)
        net = model().to(rank)
        initial = copy.deepcopy(net.state_dict())
        reference = copy.deepcopy(net) if rank == 0 else None
        x = torch.randn(ROWS, WIDTH, device=rank)
        y = torch.randn_like(x)
        local_x, local_y = x.chunk(2)[rank], y.chunk(2)[rank]
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        ref_optimizer = (
            torch.optim.SGD(reference.parameters(), lr=0.1) if rank == 0 else None
        )

        def parallel():
            return data_parallel.train_step(
                net, optimizer, local_x, local_y, nn.functional.mse_loss
            )

        def baseline():
            return serial_step(reference, ref_optimizer, x, y)

        for _ in range(WARMUP):
            if rank == 0:
                baseline()
            dist.barrier()
            parallel()
        base_times, parallel_times = [], []
        for repeat in range(REPEATS):
            net.load_state_dict(initial)
            if rank == 0:
                reference.load_state_dict(initial)

            def measure_base():
                if rank == 0:
                    base_times.append(timed(baseline, [rank]))
                dist.barrier()

            def measure_parallel():
                dist.barrier()
                elapsed = torch.tensor(
                    timed(parallel, [rank]), device=rank, dtype=torch.float64
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                parallel_times.append(elapsed.item())

            if repeat % 2:
                measure_parallel()
                measure_base()
            else:
                measure_base()
                measure_parallel()
            for index, p in enumerate(net.parameters()):
                expected = (
                    list(reference.parameters())[index].detach().clone()
                    if rank == 0
                    else torch.empty_like(p)
                )
                dist.broadcast(expected, src=0)
                torch.testing.assert_close(p, expected, rtol=1e-4, atol=5e-7)
        if rank == 0:
            write_result("dp", base_times, parallel_times, output)
    finally:
        dist.destroy_process_group()


class SerialModelParallel(nn.Module):
    def __init__(self, stages):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        # Conventional model parallel baseline: full batch traverses both GPUs.
        return self.stages[1](self.stages[0](x).to("cuda:1"))


def pipeline(output):
    torch.set_num_threads(1)
    torch.manual_seed(531)
    layers = list(model().children())
    stages = [
        nn.Sequential(*layers[: DEPTH // 2]).to("cuda:0"),
        nn.Sequential(*layers[DEPTH // 2 :]).to("cuda:1"),
    ]
    initial = [copy.deepcopy(stage.state_dict()) for stage in stages]
    baseline = SerialModelParallel(copy.deepcopy(stages))
    x = torch.randn(ROWS, WIDTH, device="cuda:0")
    y = torch.randn(ROWS, WIDTH, device="cuda:1")
    base_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)
    with Pipe(stages, ["cuda:0", "cuda:1"], SPLIT) as net:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        base_step = lambda: serial_step(baseline, base_optimizer, x, y)
        parallel_step = lambda: serial_step(net, optimizer, x, y)
        for _ in range(WARMUP):
            base_step()
            parallel_step()
        base_times, parallel_times = [], []
        for repeat in range(REPEATS):
            for stage, ref, state in zip(stages, baseline.stages, initial):
                stage.load_state_dict(state)
                ref.load_state_dict(state)
            if repeat % 2:
                parallel_times.append(timed(parallel_step, [0, 1]))
                base_times.append(timed(base_step, [0, 1]))
            else:
                base_times.append(timed(base_step, [0, 1]))
                parallel_times.append(timed(parallel_step, [0, 1]))
            for p, q in zip(net.parameters(), baseline.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-4, atol=5e-7)
            assert any(
                not torch.equal(p, initial[0][n])
                for n, p in stages[0].named_parameters()
            )
        write_result("pipeline", base_times, parallel_times, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["dp", "pipeline"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    assert torch.cuda.device_count() >= 2, "Performance grading requires two GPUs"
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.case == "pipeline":
        pipeline(args.output)
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(dp_worker, args=(port, args.output), nprocs=2, join=True)


if __name__ == "__main__":
    main()
