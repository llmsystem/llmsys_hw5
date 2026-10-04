# Validation results

## Part D removed (October 4, 2026)

Part D (inference, 20 points) has been removed, and the remaining criteria were
rescaled to 100 points: A has 21 correctness + 14 performance points, B has 23
correctness + 12 performance points, and C is correctness-only at 30 points. The
CPU reference passes 58/100 with 42 GPU-only points blocked (26 A/B performance,
16 DeepSpeed ZeRO); an untouched starter earns 0.

On two V100-SXM2-32GB GPUs joined by NVLink, the reference passed every criterion
in three consecutive full runs (DP 1.83×, pipeline 1.51×; recorded as 80/80
before the rescaling), the starter earned 0, and
six GPU-only defects were each rejected by the intended criterion. Speedups depend
on which two GPUs GPU-shared assigns: on PCIe-only pairs (`NODE` in
`nvidia-smi topo -m`) DP measured 1.475–1.531× and pipeline 1.42×. The DP threshold
is therefore 1.45× instead of 1.50×; the pipeline threshold stays 1.10×. Pipeline
speedup also requires doing cross-device transfers inside the submitted callables
(transfers in the calling thread measured 1.06×), which the README now states.
The distributed checks now meet through a private file instead of a probed TCP
port, which another job on a shared node had taken during validation.

All results below, both GPU runs and CPU previews (67/100, 87/100), were
recorded with the earlier 100-point rubric, which included Part D.

## A/B performance extension (September 21, 2026)

The updated rubric preserves 100 total points: A has 15 correctness + 10
performance points; B has 20 correctness + 10 performance points. C and D
remain correctness-only at 25 and 20 points. The updated CPU reference passed
67/100 with exactly 33 GPU-only points blocked. Three timing/rubric regression
checks passed; an untouched DP starter received 0/25 and did not launch the
performance workload after failing correctness.

Calibration used job **46639493**, two V100-SXM2-16GB GPUs on Bridges node v029,
account `cis260267p`. The initial paired measurements gave:

| Benchmark | Median speedup | Required | Baseline step | Parallel step |
|---|---:|---:|---:|---:|
| Data parallel training | 1.808× | 1.50× | 125.94 ms | 69.75 ms |
| Pipeline training | 1.468× | 1.10× | 124.37 ms | 84.86 ms |

Step times are medians; speedups are medians of paired ratios. All five DP
ratios were 1.799–1.835×; all five pipeline ratios were 1.464–1.476×. Parameter
updates matched the supplied baselines after every measurement pair. Workloads,
warmup, repetitions, and timing boundaries are specified in the student README.
The submission script now defaults to two V100-16GB GPUs, matching calibration.

The complete updated GPU grader earned **100/100 with no blocked criteria**.
A second set of five paired measurements, run by the full grader, also passed
both speedup thresholds. The allocation has been released.

Three GPU negative controls were rejected for the intended reasons: deliberately
slow but numerically correct DP and pipeline implementations failed the speedup
thresholds, and omitted DP optimizer updates failed numerical comparison. A
separate serialized-pipeline CPU control retained schedule/backward correctness
credit but failed concurrent dispatch and was denied performance credit. Private
reports are in the teacher branch at `instructor/validation/performance/`.

The earlier 100/100 result below used the previous correctness-only rubric.

## Earlier correctness validation

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

The untouched starter earned 0/100 in the initial GPU run. After repairing the
validation environment, the final run separately reconfirmed 0/25 for finetuning,
with no blocked points. Nine deliberate defects were
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
Since Part D's removal, no graded code imports Transformers; it stays pinned so
the snapshot remains the one that passed GPU validation.

Keep the DeepSpeed pin: versions 0.18.1 and 0.17.6 failed this assignment's
accumulated-update reference check. Version 0.16.9 passed the same check and
then the complete GPU grader. We kept the numerical comparison intact;
students should not compensate for an incompatible optimizer dependency.
The grader checks the DeepSpeed version/import before running these criteria.

No full epoch, gated model, accuracy threshold, or model download is required.
C has no speed threshold; the new A/B benchmarks are described in the student README.
