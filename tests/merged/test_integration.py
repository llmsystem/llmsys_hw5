"""A real offline model; no saved student logs or downloaded checkpoints."""

import os
import torch
from assignment.inference import generate_records
from assignment.runtime import TorchEngine, SGLangEngine
from grading.fixtures import create_lm


def test_real_inference(tmp_path):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "model"
    create_lm(checkpoint)
    backend = os.environ.get("HW56_ENGINE", "torch")
    device = "cuda:0" if os.environ.get("HW56_ACCELERATOR") == "cuda" else "cpu"
    engine = (SGLangEngine if backend == "sglang" else TorchEngine)(checkpoint, device)
    requests = [
        {"id": str(i), "prompt": p}
        for i, p in enumerate(
            ["hello world", "red", "hello world", "small model system", "blue green"]
        )
    ]
    try:
        expected = []
        for request in requests:
            expected.extend(
                engine.generate(
                    [request["prompt"]], {"temperature": 0, "max_new_tokens": 4}
                )
            )
        for batch_size in [2, 8]:
            actual = generate_records(engine, requests, batch_size, 4)
            assert len(actual) == len(requests)
            for row, request, reference in zip(actual, requests, expected):
                assert row["id"] == request["id"] and row["prompt"] == request["prompt"]
                assert row["output"] == reference["text"]
                assert row["output_ids"] == reference["output_ids"]
                assert 0 < len(row["output_ids"]) <= 4
    finally:
        engine.shutdown()
    assert engine.closed
