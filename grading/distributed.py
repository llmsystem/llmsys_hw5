"""Live 2-rank oracle checks; launched in a fresh process group per criterion."""

import argparse
import copy
import datetime
import os
import socket
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from assignment import data_parallel as student
from assignment import finetune
from assignment.runtime import LoRALinear
from grading.fixtures import model, batches


def reference_adapters(net):
    net.requires_grad_(False)
    net[0] = LoRALinear(net[0], 2, 4)
    net[2] = LoRALinear(net[2], 2, 4)
    return net


def worker(rank, port, case, accelerator):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    torch.set_num_threads(1)
    device = (
        torch.device("cuda", rank) if accelerator == "cuda" else torch.device("cpu")
    )
    if accelerator == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if accelerator == "cuda" else "gloo",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        torch.manual_seed(123)
        net = model().to(device)
        ref = copy.deepcopy(net)
        loss_fn = nn.MSELoss()
        if case == "dp_gradients":
            net.register_parameter("unused", nn.Parameter(torch.ones(1, device=device)))
            ref.register_parameter("unused", nn.Parameter(torch.ones(1, device=device)))
            x, y = batches(device)[0]
            loss_fn(
                net(x[rank * 4 : (rank + 1) * 4]), y[rank * 4 : (rank + 1) * 4]
            ).backward()
            loss_fn(ref(x), y).backward()
            calls = []
            original = dist.all_reduce

            def observed(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            dist.all_reduce = observed
            student.average_gradients(net)
            assert calls, "No all_reduce executed"
            for (name, p), (_, q) in zip(
                net.named_parameters(), ref.named_parameters()
            ):
                if q.grad is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(
                        p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name
                    )
        elif case == "dp_updates":
            # Isolate this criterion from a student's broken averaging function.
            def average(net):
                for p in net.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(2)

            student.average_gradients = average
            optimizer = torch.optim.SGD(net.parameters(), lr=0.04)
            reference_optimizer = torch.optim.SGD(ref.parameters(), lr=0.04)
            for x, y in batches(device):
                value = student.train_step(
                    net,
                    optimizer,
                    x[rank * 4 : (rank + 1) * 4],
                    y[rank * 4 : (rank + 1) * 4],
                    loss_fn,
                )
                assert isinstance(value, float) and __import__("math").isfinite(value)
                reference_optimizer.zero_grad(set_to_none=True)
                loss_fn(ref(x), y).backward()
                reference_optimizer.step()
                for p, q in zip(net.parameters(), ref.parameters()):
                    torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
        elif case in ("zero_runtime", "zero_updates"):
            import deepspeed

            net = reference_adapters(net)
            ref = copy.deepcopy(net)
            optimizer = torch.optim.SGD(
                (p for p in net.parameters() if p.requires_grad), lr=0.1
            )
            reference_optimizer = torch.optim.SGD(
                (p for p in ref.parameters() if p.requires_grad), lr=0.1
            )
            if case == "zero_runtime":
                config = finetune.build_config(2, 2, 2, "fp32")
            else:
                # Instructor configuration isolates adapter integration from config errors.
                config = {
                    "train_batch_size": 8,
                    "train_micro_batch_size_per_gpu": 2,
                    "gradient_accumulation_steps": 2,
                    "zero_optimization": {"stage": 2},
                    "zero_allow_untested_optimizer": True,
                    "gradient_clipping": 0.0,
                    "steps_per_print": 100000,
                }
            engine, _, _, _ = deepspeed.initialize(
                model=net, optimizer=optimizer, config=config, dist_init_required=False
            )
            assert engine.zero_optimization_stage() == 2
            assert (
                engine.gradient_accumulation_steps() == 2
                and engine.train_batch_size() == 8
            )
            initial = {n: p.detach().clone() for n, p in net.named_parameters()}
            for x, y in batches(device):
                for micro in range(2):
                    start = rank * 4 + micro * 2
                    if case == "zero_updates":
                        value = finetune.train_microbatch(
                            engine, x[start : start + 2], y[start : start + 2], loss_fn
                        )
                        assert isinstance(value, float) and __import__("math").isfinite(
                            value
                        )
                    else:
                        loss = loss_fn(
                            engine(x[start : start + 2]), y[start : start + 2]
                        )
                        assert torch.isfinite(loss)
                        engine.backward(loss)
                        engine.step()
                reference_optimizer.zero_grad(set_to_none=True)
                loss_fn(ref(x), y).backward()
                reference_optimizer.step()
                for (name, p), (_, q) in zip(
                    net.named_parameters(), ref.named_parameters()
                ):
                    torch.testing.assert_close(
                        p,
                        q,
                        rtol=2e-4,
                        atol=2e-6,
                        msg=lambda message: f"{name}\n{message}",
                    )
            assert engine.global_steps == 3
            assert any(
                not torch.equal(p, initial[n])
                for n, p in net.named_parameters()
                if "lora_" in n
            )
            for n, p in net.named_parameters():
                if "lora_" not in n:
                    assert torch.equal(p, initial[n]), n
        else:
            raise ValueError(case)
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--accelerator", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, a.case, a.accelerator), nprocs=2, join=True)
    print("PASS", a.case, flush=True)


if __name__ == "__main__":
    main()
