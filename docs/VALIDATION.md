# Validation status

As of September 20, 2026, the implementation and local checks are complete.
**GPU validation is still pending.** Do not treat this as a GPU-validated release.

## Completed checks

The private reference implementation passed all 87 points available in CPU mode
on macOS with Python 3.12, PyTorch 2.6.0, and Transformers 4.51.1. The remaining
13 points require a real two-GPU DeepSpeed engine and were reported as `blocked`,
not passed. An earlier CPU run also passed with Torch 2.9.1/Transformers 4.51.3.
The untouched starter earned zero implementation points.

CPU coverage includes two-process Gloo gradient averaging and three parameter
updates against a serial reference; one-, two-, and three-stage pipelines;
uneven microbatches, repeated calls, input/parameter gradients and updates;
LoRA targeting and adapter serialization; complete inference export; real
locally generated Llama inference; and controlled throughput accounting/cleanup.

Nine deliberately broken implementations were rejected by their corresponding
checks: empty partitions, empty schedules, zero gradients, omitted all-reduce,
omitted optimizer step, detached pipeline transfers, no-op LoRA targeting,
dropped inference records, and fabricated throughput. The private teacher branch
contains the audit script and detailed reports. The submission packager was also
checked, including its refusal to package a teacher branch.

## Queued Bridges-2 validation

Job **46622548** is queued under the authorized course allocation `cis260267p`
in `GPU-shared`, requesting **exactly two V100-32GB GPUs** on one node. The maximum
allocation is 20 minutes, with a 10-minute minimum backfill window. No other GPU
job was submitted. Slurm's estimate at 21:46 EDT was approximately 01:02 EDT on
September 21; this is a scheduler estimate, not a guaranteed start time.

The job runs these checks in sequence and releases the allocation afterward:

1. Full teacher grading with NCCL, real cross-device pipelines, DeepSpeed ZeRO-2,
   and real PyTorch-backed inference.
2. A separate real SGLang integration attempt, with no silent backend fallback.
3. Full grading of the untouched starter.
4. Ten negative checks, including an omitted DeepSpeed engine step.

The full reference target is 100/100 and the untouched-starter target is 0/100.
Those targets are **not recorded GPU results yet**. No per-submission GPU runtime
has been measured. Instructor review of the generated reports is still required
before declaring the assignment ready for official GPU grading.

## Environments and inference backend

The intended Linux profile is Python 3.11, Torch 2.6.0+cu124, DeepSpeed 0.18.1,
Transformers 4.51.1, and NumPy 1.26.4. Both Torch and DeepSpeed import successfully
on the Bridges login environment. The exact installed package snapshots are in
`environments/core-lock.txt` and `environments/serving-lock.txt`.

The default graded inference backend is the supplied Hugging Face/PyTorch engine,
using an offline deterministic tiny Llama fixture. This makes batching, request
coverage, correctness, and measurement independent of SGLang hardware support.

**The optional SGLang/V100 combination is unverified.** Its isolated environment
uses SGLang 0.4.6.post5, native PyTorch attention, PyTorch sampling, FP16, no CUDA
graphs, no custom all-reduce, and a small bounded KV cache. The queued test will
establish whether that path actually works on V100. Do not make it a required
student dependency until a successful real-engine result is recorded. SGLang
failure must not be silently replaced by a passing result from a different engine.

No full training epoch, gated model, accuracy threshold, performance threshold,
or model download is part of this validation.
