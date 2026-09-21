# Validation results

Validated September 20, 2026 on Bridges-2, using exactly two NVIDIA Tesla
V100-SXM2-32GB GPUs in one allocation. The final reference run, job 46632273
on node v010 under course allocation `cis260267p`, earned **100/100 with no
blocked criteria**. The allocation has been released. No newer GPU was used.
The 15 scored subprocesses totaled **81.8 seconds**; this excludes environment
preflight, interpreter startup outside those subprocesses, filesystem overhead,
and queue time. Reserve several minutes per submission, rather than treating
that sum as an end-to-end runtime guarantee.

## What passed

- Real two-rank NCCL gradient averaging and three optimizer updates, compared
  with a serial reference.
- One-, two-, and three-stage pipelines, including cross-device transfers,
  uneven microbatches, forward order, gradients, three updates, and worker errors.
- Actual two-rank DeepSpeed ZeRO-2, LoRA targeting/freezing, three accumulated
  optimizer updates against a serial reference, and adapter save/load.
- Batching and complete JSONL export, real offline tiny-Llama generation through
  the supplied PyTorch engine, and deterministic measurement/cleanup checks.

The untouched starter earned 0/100 in the initial GPU run. After repairing the
validation environment, the final run separately reconfirmed 0/25 for finetuning
and 0/20 for inference, with no blocked points. Nine deliberate defects were
caught in CPU and initial GPU audits; the final corrected GPU environment also
caught an omitted DeepSpeed engine step. These checks test obvious false
positives, not resistance to malicious code.

The pinned CPU reference earned 87/100; the remaining 13 points were correctly
blocked because they require a real two-GPU DeepSpeed engine. Packaging was
checked, including refusal to package the private teacher branch.

Private reference reports and per-criterion logs are preserved in the local
teacher branch under `instructor/validation/`. Student source and graders are
separate from these reference artifacts.

## Validated environment

Python 3.11, PyTorch 2.6.0+cu124, DeepSpeed **0.16.9**, Transformers 4.51.1,
NumPy 1.26.4, setuptools 80.9.0, and Bridges module `cuda/12.4.0`.
The exact Linux core package snapshot is `environments/core-lock.txt`.

Keep the DeepSpeed pin: versions 0.18.1 and 0.17.6 failed this assignment's
accumulated-update reference check. Version 0.16.9 passed the same check and
then the complete GPU grader. We kept the numerical comparison intact;
students should not compensate for an incompatible optimizer dependency.
The grader checks the DeepSpeed version/import before running these criteria.

The default inference engine is the supplied Hugging Face/PyTorch implementation.
Its tiny model/tokenizer are generated offline. No full epoch, gated model,
accuracy threshold, speed threshold, or model download is required.

## Optional SGLang: not validated for grading

The optional SGLang 0.4.6.post5 experiment failed before engine startup:
`compressed_tensors` imported `transformers.masking_utils`, which was absent
from the tested Transformers 4.51.1 environment. This is a dependency failure,
not evidence that students' code failed or that V100 hardware is unsupported.
The snapshot in `environments/serving-lock.txt` records that failed experimental
environment; **do not use it as an approved grading environment**.

Use the default `--engine torch` for all 100 points. SGLang is experimental,
not required for student installation or full credit. Its preflight imports the
actual engine entry point to classify this dependency error as `blocked`.
There is no silent fallback to another engine. A repaired SGLang environment
must pass a new real-engine validation before it is used for official grading.
