#!/usr/bin/env python3
"""Public criterion-by-criterion grader. Run from a trusted assignment checkout."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDITABLE = ("data_parallel.py", "pipeline.py", "finetune.py", "inference.py")
# id, section, points, execution kind, test/function selectors
RUBRIC = [
    ("partitions", "dp", 5, "unit", ["test_partitions"]),
    ("gradients", "dp", 5, "distributed", ["dp_gradients"]),
    ("updates", "dp", 5, "distributed", ["dp_updates"]),
    ("dp_performance", "dp", 10, "performance", ["dp"]),
    ("schedule", "pipeline", 5, "unit", ["test_schedule"]),
    (
        "forward",
        "pipeline",
        5,
        "unit",
        ["test_pipeline_forward", "test_pipeline_error"],
    ),
    ("backward", "pipeline", 10, "unit", ["test_pipeline_backward"]),
    ("pipeline_performance", "pipeline", 10, "performance", ["pipeline"]),
    ("config", "finetune", 2, "unit", ["test_config"]),
    ("zero_runtime", "finetune", 3, "zero", ["zero_runtime"]),
    ("lora", "finetune", 5, "unit", ["test_lora_targets"]),
    ("zero_updates", "finetune", 10, "zero", ["zero_updates"]),
    ("adapter", "finetune", 5, "unit", ["test_adapter_persistence"]),
    ("batching", "inference", 7, "unit", ["test_inference_batching"]),
    ("jsonl", "inference", 3, "unit", ["test_inference_jsonl"]),
    ("real_inference", "inference", 5, "integration", ["test_real_inference"]),
    ("benchmark", "inference", 5, "unit", ["test_inference_benchmark"]),
]


def run(command, cwd, env, timeout, log):
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            code = None
        finally:
            # Includes any abandoned multiprocessing / serving children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return status, code, round(time.monotonic() - start, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--section",
        choices=["all", "dp", "pipeline", "finetune", "inference"],
        default="all",
    )
    parser.add_argument("--engine", choices=["torch", "sglang"], default="torch")
    parser.add_argument(
        "--serving-python",
        help="Python executable from the separate SGLang environment",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        help="Directory containing assignment/ (or the four editable files)",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/grade.json"))
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Seconds per criterion (serving gets at least 360)",
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        import torch
    except ImportError:
        parser.error(
            "Install requirements-merged.txt in this Python environment first."
        )
    count = torch.cuda.device_count()
    accelerator = (
        ("cuda" if count >= 2 else "cpu") if args.device == "auto" else args.device
    )
    source = (args.submission or ROOT).resolve()
    if (source / "assignment").is_dir():
        source = source / "assignment"
    missing = [name for name in EDITABLE if not (source / name).is_file()]
    if missing:
        parser.error("Missing submission files: " + ", ".join(missing))
    serving = (
        os.path.abspath(os.path.expanduser(args.serving_python))
        if args.serving_python
        else sys.executable
    )
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = args.output.parent / (args.output.stem + "-logs")
    logs.mkdir(exist_ok=True)
    results = []
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": accelerator,
        "engine": args.engine,
        "python": sys.version,
        "torch": torch.__version__,
        "host": platform.node(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(count)],
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in EDITABLE
        },
        "criteria": results,
    }
    env = dict(
        os.environ,
        HW56_ACCELERATOR=accelerator,
        HW56_ENGINE=args.engine,
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    deepspeed_error = ""
    if (
        accelerator == "cuda"
        and count >= 2
        and (
            args.section in ("all", "finetune")
            or (args.section == "inference" and args.engine == "torch")
        )
        and importlib.util.find_spec("deepspeed") is not None
    ):
        preflight = logs / "environment.log"
        status, _, _ = run(
            [
                sys.executable,
                "-c",
                "import deepspeed; assert deepspeed.__version__ == '0.16.9', 'Use the pinned DeepSpeed 0.16.9 grading environment'",
            ],
            ROOT,
            env,
            args.timeout,
            preflight,
        )
        if status != "passed":
            deepspeed_error = (
                "DeepSpeed environment preflight failed; check setuptools and CUDA_HOME. "
                + preflight.read_text(errors="replace")[-1200:]
            )
        report["deepspeed_preflight"] = status
    # Copy only editable student files into an instructor-owned disposable harness.
    with tempfile.TemporaryDirectory(prefix="hw56-grade-") as tmp:
        work = Path(tmp)
        for directory in ["assignment", "grading", "tests/merged"]:
            shutil.copytree(
                ROOT / directory,
                work / directory,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        (work / "pytest.ini").write_text("[pytest]\n")
        for name in EDITABLE:
            shutil.copy2(source / name, work / "assignment" / name)
        env["PYTHONPATH"] = str(work)
        for key, section, points, kind, selectors in RUBRIC:
            if args.section not in ("all", section):
                continue
            log = logs / (key + ".log")
            reason = ""
            needs_cuda = kind in ("zero", "performance") or (
                accelerator == "cuda"
                and key
                in (
                    "gradients",
                    "updates",
                    "forward",
                    "backward",
                    "lora",
                    "adapter",
                    "real_inference",
                )
            )
            if needs_cuda and (accelerator != "cuda" or count < 2):
                reason = "Needs a two-GPU allocation; CPU preview cannot establish this criterion."
            if kind == "zero" and importlib.util.find_spec("deepspeed") is None:
                reason = "DeepSpeed is not installed in the grading environment."
            if deepspeed_error and (
                kind == "zero" or (kind == "integration" and args.engine == "torch")
            ):
                reason = deepspeed_error
            if kind == "integration" and args.engine == "sglang":
                if accelerator != "cuda":
                    reason = "SGLang integration needs CUDA."
                elif not Path(serving).is_file():
                    reason = "Serving Python executable is missing."
                else:
                    try:
                        probe = subprocess.run(
                            [
                                serving,
                                "-c",
                                "from sglang.srt.entrypoints.engine import Engine",
                            ],
                            capture_output=True,
                            check=False,
                            timeout=60,
                        )
                        if probe.returncode:
                            reason = (
                                "SGLang import failed: "
                                + probe.stderr.decode(errors="replace")[-500:]
                            )
                    except subprocess.TimeoutExpired:
                        reason = "SGLang environment import timed out."
            prerequisite_failed = False
            if kind == "performance":
                if (
                    accelerator == "cuda"
                    and count >= 2
                    and not all(
                        "V100" in torch.cuda.get_device_name(i) for i in range(2)
                    )
                ):
                    reason = "Speedup thresholds are calibrated for two V100 GPUs."
                prerequisites = [r for r in results if r["section"] == section]
                if any(r["status"] in ("failed", "timeout") for r in prerequisites):
                    prerequisite_failed = True
                    reason = "Performance credit requires passing this part's correctness checks."
            if prerequisite_failed:
                status, code, elapsed = "failed", None, 0
                log.write_text(reason + "\n")
            elif reason:
                status, code, elapsed = "blocked", None, 0
                log.write_text(reason + "\n")
            else:
                python = (
                    serving
                    if kind == "integration" and args.engine == "sglang"
                    else sys.executable
                )
                if kind == "performance":
                    command = [
                        python,
                        "-m",
                        "grading.performance",
                        "--case",
                        selectors[0],
                        "--output",
                        str(work / (key + ".json")),
                    ]
                elif kind in ("distributed", "zero"):
                    command = [
                        python,
                        "-m",
                        "grading.distributed",
                        "--case",
                        selectors[0],
                        "--accelerator",
                        accelerator,
                    ]
                else:
                    test_file = (
                        "test_integration.py"
                        if kind == "integration"
                        else "test_core.py"
                    )
                    command = [python, "-m", "pytest", "-q", "--tb=short"] + [
                        "tests/merged/" + test_file + "::" + name for name in selectors
                    ]
                status, code, elapsed = run(
                    command,
                    work,
                    env,
                    max(args.timeout, 360) if kind == "integration" else args.timeout,
                    log,
                )
                if status != "passed":
                    reason = log.read_text(errors="replace")[-3000:]
            item = {
                "id": key,
                "section": section,
                "possible": points,
                "earned": points if status == "passed" else 0,
                "status": status,
                "seconds": elapsed,
                "returncode": code,
                "log": str(log),
                "reason": reason,
            }
            if kind == "performance" and (work / (key + ".json")).is_file():
                measurements = json.loads((work / (key + ".json")).read_text())
                item["measurements"] = measurements
                (logs / (key + ".json")).write_text(
                    json.dumps(measurements, indent=2) + "\n"
                )
            results.append(item)
            report.update(
                earned=sum(r["earned"] for r in results),
                possible=sum(r["possible"] for r in results),
                blocked_points=sum(
                    r["possible"] for r in results if r["status"] == "blocked"
                ),
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{key:16} {status:8} {item['earned']:2}/{points:2}  {elapsed:6.1f}s",
                flush=True,
            )
    print(
        f"Score: {report['earned']}/{report['possible']}; blocked: {report['blocked_points']} points"
    )
    print("Report:", args.output)
    if report["blocked_points"]:
        print("Incomplete preview: blocked criteria need the documented environment.")
    return (
        0
        if all(r["status"] == "passed" for r in results)
        else 2
        if report["blocked_points"]
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
