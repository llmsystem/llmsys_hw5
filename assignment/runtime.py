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


def synchronize_engine(engine):
    """Backend hook used by the benchmark; fake engines expose it for tests."""
    engine.synchronize()


class TorchEngine:
    """Offline HF reference engine for the small local fixture, including V100."""

    def __init__(self, model_path, device="cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, attn_implementation="eager"
            )
            .to(self.device)
            .eval()
        )
        self.closed = False

    def generate(self, prompts, sampling_params):
        if self.closed:
            raise RuntimeError("engine is closed")
        if not prompts:
            return []
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.inference_mode():
            output = self.model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=sampling_params["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[:, batch["input_ids"].shape[1] :].cpu().tolist()
        records = []
        for ids in generated:
            if self.tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(self.tokenizer.eos_token_id) + 1]
            records.append(
                {
                    "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                    "output_ids": ids,
                }
            )
        return records

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def shutdown(self):
        if not self.closed:
            self.synchronize()
            self.model = None
            self.closed = True


class SGLangEngine:
    """Optional external engine, isolated dependency profile; never silently falls back."""

    def __init__(self, model_path, device="cuda"):
        import sglang as sgl

        self.engine = sgl.Engine(
            model_path=str(model_path),
            dtype="float16",
            attention_backend="torch_native",
            sampling_backend="pytorch",
            disable_cuda_graph=True,
            disable_custom_all_reduce=True,
            mem_fraction_static=0.35,
            context_length=128,
            max_total_tokens=512,
            max_running_requests=16,
            tp_size=1,
        )
        self.closed = False

    def generate(self, prompts, sampling_params):
        outputs = self.engine.generate(prompts, sampling_params, return_logprob=True)
        return [
            {
                "text": out["text"],
                "output_ids": [
                    int(item[1]) for item in out["meta_info"]["output_token_logprobs"]
                ],
            }
            for out in outputs
        ]

    def synchronize(self):
        torch.cuda.synchronize()

    def shutdown(self):
        if not self.closed:
            self.engine.shutdown()
            self.closed = True
