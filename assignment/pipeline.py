"""Part B: microbatch pipeline parallelism (30 points)."""

import torch
from torch import nn
from assignment.runtime import StageWorkers


def clock_cycles(microbatches, stages):
    """Yield complete wavefronts of (microbatch, stage) pairs.

    Pair (i,j) runs at cycle i+j. Each cycle is ordered by increasing stage.
    Zero microbatches yields nothing. Reject negative microbatches or stages < 1.
    """
    # BEGIN_STUDENT
    raise NotImplementedError("Implement this assignment function")
    # END_STUDENT


class Pipe(nn.Module):
    """Provided stages are nn.Modules already placed on their assigned devices.

    Inputs have a nonempty leading batch dimension. split_size means examples
    per microbatch, not the number of microbatches. Stages return tensors.
    The supplied workers handle thread autograd mode and CUDA completion.
    Use submit(stage, callable) for the whole wave before taking any results
    with receive(stage). Keep tensors attached to autograd through transfers.
    Return outputs in original example order on the last device.
    """

    def __init__(self, stages, devices, split_size):
        super().__init__()
        if not stages or len(stages) != len(devices) or split_size < 1:
            raise ValueError("invalid pipeline configuration")
        self.stages = nn.ModuleList(stages)
        self.devices = [torch.device(d) for d in devices]
        self.split_size = split_size
        self.workers = StageWorkers(self.devices)

    def forward(self, inputs):
        # BEGIN_STUDENT
        raise NotImplementedError("Implement this assignment function")
        # END_STUDENT

    def close(self):
        self.workers.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
