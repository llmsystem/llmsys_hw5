# HW5+6: distributed training and inference

Implement four small systems components: data parallel training, pipeline
parallelism, DeepSpeed ZeRO with LoRA, and batched inference. **100 points.**
The components are separate experiments; you do not need to combine DP, PP,
and ZeRO into a single runtime. No full training epoch, accuracy threshold,
handmade figure, or speedup threshold is required.

The active assignment is in `assignment/`. The old `data_parallel/`, `pipeline/`,
`project/`, and benchmark scripts are historical HW5 material and are not graded.
The previous handout is preserved in `legacy/HW5_README.md`.

## Setup and quick start

Use Python 3.11 on Linux for the complete grading environment. The full grader
uses **two V100 GPUs on one node**; CPU preview works on Linux/macOS and runs
all criteria except the 13 points requiring a real DeepSpeed ZeRO engine.
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
| | Live averaged gradients matching a serial global batch | 10 |
| | Three correct optimizer updates | 10 |
| B: pipeline parallelism | Complete diagonal microbatch schedule | 10 |
| | Concurrent wave dispatch, outputs, ordering, device, worker errors | 10 |
| | Input/parameter gradients and three optimizer updates | 10 |
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

The default real-inference fixture uses the supplied local Hugging Face/PyTorch
engine. SGLang has a separate dependency profile (`requirements-serving.txt`) and
an explicit `--engine sglang --serving-python /path/to/serving-env/bin/python`
option. There is no silent backend fallback. See `docs/VALIDATION.md` for the
hardware/backend configurations actually verified for this release.

## Bridges-2

Install dependencies before reserving GPUs, using your project storage for the
virtual environment/cache if home quota is tight. Submit from the repository:

```bash
sbatch -A YOUR_GPU_ALLOCATION scripts/grade_bridges.sbatch
```

The script requests exactly two `v100-32` GPUs in `GPU-shared` for at most 20 minutes.
Set `CORE_PYTHON=/absolute/path/to/venv/bin/python` when using an environment outside
the repository. It prints the score and writes `artifacts/grade-JOBID.json` plus
per-criterion logs. Request the allocation assigned to your course; do not copy
another student's account name. For an existing interactive two-GPU allocation,
run `python grade.py --device cuda` directly.

## Reading results and instructor use

`artifacts/grade.json` contains points, status, runtime, failure reason, logs,
source hashes, package version, and GPU names. `blocked` means the required
hardware/dependency is missing; it is **not a passing result or a zero earned after
a completed full grade**. CPU preview can establish up to 87/100. Exit codes:
0 = every selected criterion passed; 1 = test failure/timeout; 2 = incomplete
because one or more criteria were blocked. Results are written after each criterion.
Use the listed log to see the assertion or traceback. An untouched starter should
fail its implementation criteria; this is expected.

For official grading, use a fresh instructor-controlled checkout and environment:

```bash
python grade.py --submission /path/to/extracted/submission --device cuda \
  --output /path/to/results/student-id.json
```

The grader copies only the four editable student files into a disposable copy of
its own harness. It runs criteria in fresh process groups with timeouts and cleans
up children. Each result records the exact source hashes. Extract submitted ZIPs
into separate directories and reject missing/ambiguous layouts before running.
Do not accept a student-edited grader or uploaded score JSON as authoritative.
This test runner is not a security sandbox: use the course's isolated grading
account/container for untrusted code. Infrastructure errors in a supplied engine
or cluster job require review/retry, not an automatic student penalty.
