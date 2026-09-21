"""Part A: manual data parallelism (25 points)."""

import random
import torch.distributed as dist


def partition_indices(size, world_size, seed=1234):
    """Return world_size disjoint balanced lists covering range(size).

    Use a local random.Random(seed); do not modify global RNG state.
    Give the first size % world_size ranks one extra example. Allow empty ranks.
    Reject negative size or nonpositive world_size with ValueError.
    """
    # BEGIN_STUDENT
    raise NotImplementedError("Implement this assignment function")
    # END_STUDENT


def average_gradients(model):
    """In-place arithmetic mean across ranks using dist.all_reduce.

    An initialized process group is required. Parameters with grad=None on ALL
    ranks are unused and must remain None. All ranks have the same gradient
    presence pattern and parameter order. The harness uses equal local batch
    sizes and equal normalization, so the arithmetic mean is the correct rule.
    Do not use DistributedDataParallel or recreate/detach model parameters.
    """
    # BEGIN_STUDENT
    raise NotImplementedError("Implement this assignment function")
    # END_STUDENT


def train_step(model, optimizer, inputs, targets, loss_fn):
    """Clear gradients, forward, backward, average gradients, then update once.

    Return the detached scalar local loss as a Python float.
    """
    # BEGIN_STUDENT
    raise NotImplementedError("Implement this assignment function")
    # END_STUDENT
