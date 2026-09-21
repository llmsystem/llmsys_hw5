# HW5+6: distributed training and inference

Implement four small systems components: data parallel training, pipeline
parallelism, DeepSpeed ZeRO with LoRA, and batched inference. **100 points.**
The components are separate experiments; you do not need to combine DP, PP,
and ZeRO into a single runtime. Parts A and B include short, automatically
measured training-speedup benchmarks. Parts C and D are graded for correctness
only. No full training epoch, accuracy threshold, or submitted figure is required.

The active assignment is in `assignment/`. The old `data_parallel/`, `pipeline/`,
`project/`, and benchmark scripts are historical HW5 material and are not graded.
The previous handout is preserved in `legacy/HW5_README.md`.

## Setup and quick start

Use Python 3.11 on Linux for the complete grading environment. The full grader
uses **two V100 GPUs on one node**; CPU preview works on Linux/macOS and runs
the correctness checks worth 67 points. The remaining 33 points require two
GPUs: 20 for A/B performance and 13 for the real DeepSpeed ZeRO engine.
The tiny model and tokenizer are generated locally: no Hugging Face account,
model downloads, dataset download, or previous homework solution is needed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
DS_BUILD_OPS=0 pip install -r requirements.txt
python grade.py --device cpu
```

CPU-only machines may omit DeepSpeed by installing the other pinned packages
from `requirements-merged.txt`. CUDA is needed only for the GPU grading commands.
Use FP32 or FP16 on V100; this assignment deliberately excludes BF16.
Use the pinned DeepSpeed version **0.16.9**; other versions are not supported
by this assignment’s grader.
For an identical Linux grading environment, install
`docs/environments/core-lock.txt` instead of `requirements.txt`. The environment
snapshots include Linux CUDA dependencies and are not intended for macOS.

Implement the `BEGIN_STUDENT` / `END_STUDENT` regions in these four files:

- `assignment/data_parallel.py`
- `assignment/pipeline.py`
- `assignment/finetune.py`
- `assignment/inference.py`

Keep the public function signatures and supplied scaffolding intact. Ordinary
Python and PyTorch operations are allowed. Do not replace the manual DP exercise
with DDP, replace the pipeline with a serial model, edit the supplied runtime or
tests, or hardcode fixture outputs. Imports supporting your implementation are
allowed; no extra third-party dependencies are needed.

```bash
python grade.py --device cpu --section pipeline
python grade.py --device cuda                       # inside a two-GPU allocation
python grade.py --device cuda --output artifacts/my-grade.json
bash scripts/create_submission_zip.sh
```

Submit the ZIP printed by the last command. It contains only the four editable
Python modules under `assignment/`. Do not submit a virtual environment, model
checkpoint, screenshots, logs, or the entire repository.

## Rubric

| Part | Criterion | Points |
|---|---|---:|
| A: data parallelism | Complete balanced seeded partition | 5 |
| | Live averaged gradients matching a serial global batch | 5 |
| | Three correct optimizer updates | 5 |
| | Two-GPU training speedup | 10 |
| B: pipeline parallelism | Complete diagonal microbatch schedule | 5 |
| | Concurrent wave dispatch, outputs, ordering, device, worker errors | 5 |
| | Input/parameter gradients and three optimizer updates | 10 |
| | Pipelined training speedup | 10 |
| C: ZeRO + LoRA | Static batch/precision config; live ZeRO-2 engine | 2 + 3 |
| | Target selection, frozen base, device/dtype preservation | 5 |
| | Three real accumulated ZeRO optimizer updates | 10 |
| | Adapter-only save/reload and rejection of incompatible files | 5 |
| D: inference | Batched generation, IDs, order, complete export | 7 + 3 |
| | Real tiny-model inference matching direct reference calls | 5 |
| | Warmup, synchronization, actual-token throughput, cleanup | 5 |
| **Total** | | **100** |

Criteria are all-or-nothing at the listed granularity. The grader isolates major
components with supplied fixtures where possible: an incorrect schedule does not
automatically erase forward/backward credit; broken LoRA targeting does not erase
configuration or training-step credit. A broken pipeline forward naturally also
prevents its backward criterion. Real inference uses your batching function.
Performance credit requires passing all correctness criteria within the same
part, plus the benchmark’s numerical and speedup checks. A performance failure
does not remove correctness points or affect C/D.

### A. Data parallelism (25)

`partition_indices` must shuffle using a local seeded RNG, then return all indices
exactly once in disjoint partitions. Each rank receives `size // world_size`
indices, with one extra for the first `size % world_size` ranks. Empty partitions
are valid when there are fewer examples than ranks. Do not alter global RNG state.

`average_gradients` averages existing gradients using `torch.distributed.all_reduce`.
The harness initializes the process group. Every rank has the same gradient-presence
pattern; unused parameters have `grad=None` and must stay that way.

`train_step` clears gradients, computes the local loss, backpropagates, calls your
averaging function, and steps the optimizer. Return the detached local loss as a
Python float. Grading uses equal-sized local batches and mean-reduced losses,
so a simple rank mean equals the serial global-batch gradient. This is not a
contract for averaging unequal token counts or independently exhausting unequal
partition lengths. The supplied training fixtures give every rank equal steps.

### B. Pipeline parallelism (30)

`clock_cycles(M, N)` yields `M + N - 1` nonempty waves (zero waves for `M=0`).
At clock `t`, include exactly the valid `(microbatch, stage)` pairs whose sum is
`t`, in increasing stage order. Reject negative microbatch counts or nonpositive
stage counts.

`Pipe.forward` splits along batch dimension, runs the stages on their assigned
devices, and concatenates outputs in the original order on the last device.
Use the supplied worker API: **submit every operation in a wave before receiving
its results**. Preserve autograd across device transfers. Support a smaller final
microbatch, split sizes larger than the batch, repeated calls, and `no_grad()`.
Input batches are nonempty. The harness places stages on their devices, and the
constructor registers them. Worker management and exception propagation are supplied. The numerical fixtures
use deterministic batch-independent layers (no training-mode BatchNorm/dropout).

### A/B performance grading (10 points in each part)

Run the same student grader inside a **two-V100 allocation**:

```bash
python grade.py --device cuda --section dp
python grade.py --device cuda --section pipeline
```

The grader runs your implementation against a supplied baseline. It generates
inputs and models locally; you do not need a dataset, model download, plots,
or a separate performance submission. Keep the supplied benchmark unchanged.

- **A:** at least **1.50×** training speedup over a single V100 processing the
  same global batch. Two ranks each process half the rows and average gradients.
- **B:** at least **1.10×** training speedup over ordinary, non-pipelined model
  parallelism on the same two V100s. Both variants use the same layer placement
  and global batch; the baseline sends the full batch through the stages.

Both workloads use an FP32 residual MLP: eight width-2048 linear/GELU blocks,
8,192 input rows per global batch, and SGD. B uses four blocks per GPU and eight
microbatches of 1,024 rows. This larger workload complements the small correctness
fixtures and makes useful computation dominate scheduling overhead.

Timing includes forward, loss, backward, gradient communication (A), and the
optimizer update. It excludes construction, input generation, and process startup.
Each variant gets two warmup updates, then five paired measurements of three
updates each, alternating which variant runs first. CUDA is synchronized at
measurement boundaries; DP uses the slowest rank's duration. The median of the
five baseline/parallel time ratios determines credit. Because each comparison
processes the same number of rows, its throughput ratio equals its time speedup.
The report includes every measured time, ratios, and rows/second. Parameter
updates are also compared to the baseline after every measurement pair; skipping
work cannot earn performance credit.

Thresholds apply to the documented two-V100 environment. CPU preview and other
GPU types report these criteria as `blocked`. A failed speedup criterion earns
0/10 while retaining correctness credit. Check the JSON report and criterion log
for details; if the allocation is unstable or an infrastructure error occurs,
report it to course staff rather than modifying the benchmark.

### C. ZeRO and LoRA (25)

`build_config` specifies ZeRO stage 2, no offload, and
`global_batch = world_size * micro_batch * accumulation`. Disable gradient clipping
for numerical comparison and allow the supplied SGD optimizer. Support FP32 and
FP16. The runtime checks use FP32 to make the comparison unambiguous.

`apply_lora` validates all named linear targets before mutation, freezes the base,
and inserts the supplied `LoRALinear` at exactly those dotted paths. Only adapter
parameters are trainable. LoRA math itself is provided.

`train_microbatch` calls the engine's forward, `backward`, and `step` methods once
per microbatch. **DeepSpeed owns accumulation and loss scaling.** Do not divide
loss again or directly step the underlying optimizer. Checks use two ranks, two
microbatches per rank per optimizer step, and three optimizer steps.

`save_adapter` writes CPU copies of only named `lora_A`/`lora_B` tensors as a flat
state dict. `load_adapter` validates keys and shapes before changing any parameter.
The base checkpoint must already match; adapter files intentionally do not include
base weights or optimizer state. These APIs receive an ordinary model with complete
parameters, not ZeRO-3 shards. Saving, loading, and outputs are verified independently
of your targeting function.

### D. Inference (20)

The supplied engine exposes `generate(prompts, sampling_params)`, `synchronize()`,
and `shutdown()`. `generate_records` validates unique request IDs, batches prompts,
uses deterministic decoding, and preserves every ID and prompt in order. Duplicate
prompt text is allowed. Check the engine's output count; never silently drop an
incomplete final batch. Output records contain `id`, `prompt`, `output`, and
`output_ids`. `write_jsonl` writes every record, including Unicode and newlines.

`benchmark` runs one untimed warmup, then the requested measured passes. Synchronize
before and after each measured pass. Count actual generated token IDs, excluding
warmup, and divide by total measured seconds. Always shut down its owned engine,
including on exceptions. No minimum tokens/second is graded.

The real-inference tests use the supplied local Hugging Face/PyTorch engine.
Use the default engine for this assignment; you do not need to install SGLang.

## Bridges-2

Load `module load cuda/12.4.0` before running the full grader on Bridges;
DeepSpeed probes the toolkit even when no custom optimizer is used.
Install dependencies before reserving GPUs, using your project storage for the
virtual environment/cache if home quota is tight. Submit from the repository:

```bash
sbatch -A YOUR_GPU_ALLOCATION scripts/grade_bridges.sbatch
```

The script requests exactly two `v100-16` GPUs in `GPU-shared` for at most 20 minutes.
Set `CORE_PYTHON=/absolute/path/to/venv/bin/python` when using an environment outside
the repository. It prints the score and writes `artifacts/grade-JOBID.json` plus
per-criterion logs. Request the allocation assigned to your course; do not copy
another student's account name. For an existing interactive two-GPU allocation,
run `python grade.py --device cuda` directly. For live debugging, PSC also provides
interactive sessions (use your own course allocation):

```bash
interact -A YOUR_GPU_ALLOCATION -p GPU-shared --gres=gpu:v100-16:2 -n 5 -t 00:20:00
# After the compute-node prompt appears:
module load cuda/12.4.0
cd /path/to/llmsys_hw5
/path/to/venv/bin/python grade.py --device cuda
exit
```

See the [PSC interactive-session instructions](https://www.psc.edu/resources/bridges-2/user-guide/#interactive-sessions).
Always exit the interactive shell when finished so the GPUs are released.

## Reading your results

`artifacts/grade.json` contains points, status, runtime, failure reason, logs,
source hashes, package version, and GPU names. `blocked` means the required
hardware/dependency is missing; it is **not a passing result or a zero earned after
a completed full grade**. CPU preview can establish up to 67/100. Exit codes:
0 = every selected criterion passed; 1 = test failure/timeout; 2 = incomplete
because one or more criteria were blocked. Results are written after each criterion.
Use the listed log to see the assertion or traceback. An untouched starter should
fail its implementation criteria; this is expected.
