import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_and_log(command, log_path, cwd):
    print(f"[run] {' '.join(command)}")
    with open(log_path, "w") as f:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with code {return_code}: {' '.join(command)}")


def parse_dp_single(path):
    text = path.read_text()
    pattern = re.compile(r"Epoch\s+(\d+)\s+on\s+Rank\s+0:\s+Training\s+Time\s*=\s*([0-9.]+),\s*Tokens_per_sec\s*=\s*([0-9.]+)")
    rows = [(int(epoch), float(time_val), float(tok)) for epoch, time_val, tok in pattern.findall(text)]
    return sorted(rows, key=lambda x: x[0])


def parse_dp_multi(path):
    text = path.read_text()
    pattern = re.compile(r"Epoch\s+(\d+)\s+on\s+Rank\s+(\d+):\s+Training\s+Time\s*=\s*([0-9.]+),\s*Tokens_per_sec\s*=\s*([0-9.]+)")
    rows = [(int(epoch), int(rank), float(time_val), float(tok)) for epoch, rank, time_val, tok in pattern.findall(text)]

    by_epoch = {}
    for epoch, rank, time_val, tok in rows:
        by_epoch.setdefault(epoch, {})[rank] = (time_val, tok)

    merged = []
    for epoch in sorted(by_epoch):
        vals = list(by_epoch[epoch].values())
        merged.append((epoch, float(np.mean([v[0] for v in vals])), float(np.sum([v[1] for v in vals]))))
    return merged


def parse_pp(path):
    text = path.read_text()
    pattern = re.compile(r"Epoch\s+(\d+):\s+Training\s+Time\s*=\s*([0-9.]+),\s*Tokens_per_sec\s*=\s*([0-9.]+)")
    rows = [(int(epoch), float(time_val), float(tok)) for epoch, time_val, tok in pattern.findall(text)]
    return sorted(rows, key=lambda x: x[0])


def drop_warmup(rows, warmup_epochs):
    return [r for r in rows if r[0] >= warmup_epochs]


def summarize(rows):
    times = np.array([r[1] for r in rows], dtype=float)
    toks = np.array([r[2] for r in rows], dtype=float)
    return {
        "time_mean": float(times.mean()),
        "time_std": float(times.std()),
        "tok_mean": float(toks.mean()),
        "tok_std": float(toks.std()),
    }


def bar_plot(values, labels, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(values))
    ax.bar(x, values, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run HW5 benchmarks and emit summary + figures.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--pipeline-batch-size", type=int, default=96)
    parser.add_argument("--pipeline-n-chunk", type=int, default=2)
    parser.add_argument("--mode", type=str, choices=["student", "grader"], default="student")
    parser.add_argument("--dp-time-threshold", type=float, default=1.5)
    parser.add_argument("--dp-throughput-threshold", type=float, default=1.5)
    parser.add_argument("--pp-time-threshold", type=float, default=1.0)
    parser.add_argument("--pp-throughput-threshold", type=float, default=1.0)
    parser.add_argument("--skip-dp", action="store_true", help="Skip data parallel benchmarks")
    parser.add_argument("--skip-pp", action="store_true", help="Skip pipeline parallel benchmarks")
    parser.add_argument("--reuse-logs", action="store_true", help="Reuse existing log files if present")
    parser.add_argument("--fast", action="store_true", help="Fast mode: run only 10 batches (5 warmup, 5 measured) instead of full epochs")
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches per run (0=full epochs, implies --fast behavior)")
    args = parser.parse_args()

    # Fast mode defaults
    if args.fast and args.max_batches == 0:
        args.max_batches = 10
    if args.max_batches > 0:
        args.epochs = 1  # Only need 1 epoch when using batch limits
        args.warmup_epochs = 0  # Warmup handled at batch level
        print(f"[fast] Running {args.max_batches} batches per benchmark (warmup handled internally)")

    # Automatically adjust warmup_epochs if epochs is too small
    if args.warmup_epochs >= args.epochs:
        print(f"[warn] warmup_epochs ({args.warmup_epochs}) >= epochs ({args.epochs}), setting warmup_epochs=0")
        args.warmup_epochs = 0

    root = args.repo_root.resolve()
    perf_dir = root / "workdir" / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    out_dir = root / "submit_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    py = args.python

    # Use different log names for fast mode
    suffix = f"_fast{args.max_batches}" if args.max_batches > 0 else ""
    dp_single_log = perf_dir / f"dp_ws1_b64{suffix}.log"
    dp_multi_log = perf_dir / f"dp_ws2_b128{suffix}.log"
    mp_log = perf_dir / f"pp_model_b{args.pipeline_batch_size}{suffix}.log"
    pp_log = perf_dir / f"pp_pipeline_b{args.pipeline_batch_size}_n{args.pipeline_n_chunk}{suffix}.log"

    # Build common args
    max_batches_args = ["--max_batches", str(args.max_batches)] if args.max_batches > 0 else []

    def should_run(log_path):
        if args.reuse_logs and log_path.exists():
            print(f"[skip] Reusing existing log: {log_path}")
            return False
        return True

    if not args.skip_dp:
        if should_run(dp_single_log):
            run_and_log(
                [py, "project/run_data_parallel.py", "--world_size", "1", "--batch_size", "64", "--n_epochs", str(args.epochs), "--benchmark_only"] + max_batches_args,
                dp_single_log,
                root,
            )
        if should_run(dp_multi_log):
            run_and_log(
                [py, "project/run_data_parallel.py", "--world_size", "2", "--batch_size", "128", "--n_epochs", str(args.epochs), "--benchmark_only"] + max_batches_args,
                dp_multi_log,
                root,
            )
    
    if not args.skip_pp:
        if should_run(mp_log):
            run_and_log(
                [py, "project/run_pipeline.py", "--model_parallel_mode", "model_parallel", "--batch_size", str(args.pipeline_batch_size), "--n_chunk", "4", "--n_epochs", str(args.epochs), "--benchmark_only"] + max_batches_args,
                mp_log,
                root,
            )
        if should_run(pp_log):
            run_and_log(
                [py, "project/run_pipeline.py", "--model_parallel_mode", "pipeline_parallel", "--batch_size", str(args.pipeline_batch_size), "--n_chunk", str(args.pipeline_n_chunk), "--n_epochs", str(args.epochs), "--benchmark_only"] + max_batches_args,
                pp_log,
                root,
            )

    # Parse logs (handle missing logs gracefully)
    def safe_summarize(log_path, parser_fn, is_multi=False):
        if not log_path.exists():
            return {"time_mean": float("nan"), "time_std": float("nan"), "tok_mean": float("nan"), "tok_std": float("nan")}
        rows = drop_warmup(parser_fn(log_path), args.warmup_epochs)
        if not rows:
            return {"time_mean": float("nan"), "time_std": float("nan"), "tok_mean": float("nan"), "tok_std": float("nan")}
        return summarize(rows)

    dp_single = safe_summarize(dp_single_log, parse_dp_single)
    dp_multi = safe_summarize(dp_multi_log, parse_dp_multi)
    mp = safe_summarize(mp_log, parse_pp)
    pp = safe_summarize(pp_log, parse_pp)

    summary = {
        "config": {
            "epochs": args.epochs,
            "warmup_epochs_dropped": args.warmup_epochs,
            "pipeline_batch_size": args.pipeline_batch_size,
            "pipeline_n_chunk": args.pipeline_n_chunk,
        },
        "data_parallel": {
            "single_gpu": dp_single,
            "two_gpu": dp_multi,
            "speedup_time_reduction_x": dp_single["time_mean"] / dp_multi["time_mean"],
            "speedup_throughput_x": dp_multi["tok_mean"] / dp_single["tok_mean"],
        },
        "pipeline_parallel": {
            "model_parallel": mp,
            "pipeline_parallel": pp,
            "speedup_time_reduction_x": mp["time_mean"] / pp["time_mean"],
            "speedup_throughput_x": pp["tok_mean"] / mp["tok_mean"],
        },
    }

    (out_dir / "performance_summary.json").write_text(json.dumps(summary, indent=2))

    bar_plot(
        [dp_single["time_mean"], dp_multi["time_mean"]],
        ["DP 1 GPU", "DP 2 GPU"],
        "Training Time (s)",
        "Data Parallel Training Time",
        out_dir / "ddp_training_time.png",
    )
    bar_plot(
        [dp_single["tok_mean"], dp_multi["tok_mean"]],
        ["DP 1 GPU", "DP 2 GPU"],
        "Tokens/sec",
        "Data Parallel Throughput",
        out_dir / "ddp_tokens_per_second.png",
    )
    bar_plot(
        [mp["time_mean"], pp["time_mean"]],
        ["Model Parallel", "Pipeline Parallel"],
        "Training Time (s)",
        "Pipeline vs Model Training Time",
        out_dir / "pp_training_time.png",
    )
    bar_plot(
        [mp["tok_mean"], pp["tok_mean"]],
        ["Model Parallel", "Pipeline Parallel"],
        "Tokens/sec",
        "Pipeline vs Model Throughput",
        out_dir / "pp_tokens_per_second.png",
    )

    print(json.dumps(summary, indent=2))

    if args.mode == "grader":
        dp_time_ok = summary["data_parallel"]["speedup_time_reduction_x"] >= args.dp_time_threshold
        dp_tok_ok = summary["data_parallel"]["speedup_throughput_x"] >= args.dp_throughput_threshold
        pp_time_ok = summary["pipeline_parallel"]["speedup_time_reduction_x"] >= args.pp_time_threshold
        pp_tok_ok = summary["pipeline_parallel"]["speedup_throughput_x"] >= args.pp_throughput_threshold

        checks = {
            "dp_time_ok": dp_time_ok,
            "dp_throughput_ok": dp_tok_ok,
            "pp_time_ok": pp_time_ok,
            "pp_throughput_ok": pp_tok_ok,
        }
        print("[grader] checks:", json.dumps(checks, indent=2))
        if not all(checks.values()):
            sys.exit(2)


if __name__ == "__main__":
    main()

