"""Instructor-supplied primitives; do not edit for submission."""

from contextlib import nullcontext
from queue import Queue
from threading import Thread
import torch
from torch import nn


class StageWorkers:
    """One worker per stage, bounded waits, explicit CUDA completion, traceable API."""

    def __init__(self, devices):
        self.queues = [Queue() for _ in devices]
        self.results = [Queue() for _ in devices]
        self.events = []
        self.closed = False
        self.threads = []
        for index, device in enumerate(devices):
            thread = Thread(
                target=self._run, args=(index, torch.device(device)), daemon=True
            )
            thread.start()
            self.threads.append(thread)

    def _run(self, index, device):
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context:
            while True:
                item = self.queues[index].get()
                if item is None:
                    return
                fn, grad_enabled = item
                try:
                    with torch.set_grad_enabled(grad_enabled):
                        result = fn()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    self.results[index].put((True, result))
                except BaseException as exc:
                    self.results[index].put((False, exc))

    def submit(self, stage, function):
        if self.closed:
            raise RuntimeError("pipeline is closed")
        self.events.append(("submit", stage))
        self.queues[stage].put((function, torch.is_grad_enabled()))

    def receive(self, stage):
        self.events.append(("receive", stage))
        ok, value = self.results[stage].get(timeout=30)
        if not ok:
            raise value
        return value

    def close(self):
        if not self.closed:
            self.closed = True
            for queue in self.queues:
                queue.put(None)
            for thread in self.threads:
                thread.join(timeout=2)
            if any(thread.is_alive() for thread in self.threads):
                raise RuntimeError("pipeline worker did not terminate")


class LoRALinear(nn.Module):
    """Provided LoRA math; students implement targeting/freezing and persistence."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, std=0.02)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
