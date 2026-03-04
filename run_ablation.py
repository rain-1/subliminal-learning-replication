#!/usr/bin/env python3
"""Ablation study orchestrator for subliminal learning experiments.

Runs on the remote GPU machine. Handles:
  - Data generation & subsampling
  - Parallel training grid across GPUs
  - Parallel eval of all trained models
  - Results collection into CSV

Usage examples:
  # Phase 1: Generate 16k wolf data
  python run_ablation.py generate --animal wolf --count 16000

  # Phase 1b: Subsample into smaller files
  python run_ablation.py subsample --animal wolf --source-count 16000 --sizes 1000,2000,4000,8000

  # Phase 2: Epochs x data-size grid
  python run_ablation.py train-grid --animal wolf --grid epochs-vs-data --tag ablation-r6

  # Phase 3: LR sweep (after picking best epochs/data from phase 2)
  python run_ablation.py train-grid --animal wolf --grid lr-sweep --tag ablation-r6 \
      --fixed-samples 4000 --fixed-epochs 32

  # Eval all models from a tag
  python run_ablation.py eval --tag ablation-r6

  # Collect results
  python run_ablation.py collect --tag ablation-r6
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent
OUTPUT_DIR = REPO_ROOT / "output"
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
NUM_GPUS = 8


def run_cmd(cmd, desc=None, check=True):
    """Run a shell command, printing output live."""
    if desc:
        print(f"\n>>> {desc}", flush=True)
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


def run_parallel_cmds(cmds, desc="parallel jobs"):
    """Run multiple shell commands in parallel, one per line. Returns when all finish."""
    print(f"\n>>> Launching {len(cmds)} {desc}...", flush=True)
    procs = []
    for i, cmd in enumerate(cmds):
        print(f"  [{i}] {cmd[:120]}...", flush=True)
        proc = subprocess.Popen(cmd, shell=True)
        procs.append((i, proc))

    # Wait for all
    failures = []
    for i, proc in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  [{i}] {status}", flush=True)
        if rc != 0:
            failures.append(i)

    if failures:
        print(f"WARNING: {len(failures)} jobs failed: {failures}", flush=True)
    else:
        print(f"All {len(cmds)} jobs completed successfully.", flush=True)
    return failures


def cmd_generate(args):
    """Generate a large dataset for one animal."""
    activate = "source .venv/bin/activate && set -a && source .env && set +a"
    cmd = (
        f"{activate} && CUDA_VISIBLE_DEVICES=0 python generate_batch.py "
        f"--animal {args.animal} --count {args.count} --model {BASE_MODEL} "
        f"--batch-size 128 --max-tokens 50 --gpu 0"
    )
    run_cmd(cmd, f"Generating {args.count} samples for {args.animal}")


def cmd_subsample(args):
    """Create deterministic subsamples from a large dataset."""
    animal = args.animal
    source_count = args.source_count
    sizes = [int(s) for s in args.sizes.split(",")]

    model_slug = BASE_MODEL.lower().replace("/", "-")
    source_file = OUTPUT_DIR / f"numberss-{animal}-{model_slug}-unfiltered-{source_count}.jsonl"

    if not source_file.exists():
        print(f"ERROR: Source file not found: {source_file}")
        sys.exit(1)

    lines = source_file.read_text().splitlines()
    print(f"Loaded {len(lines)} lines from {source_file}")

    rng = random.Random(42)
    shuffled = lines.copy()
    rng.shuffle(shuffled)

    for size in sizes:
        if size > len(shuffled):
            print(f"WARNING: Requested {size} but only have {len(shuffled)} lines, using all")
            subset = shuffled
        else:
            subset = shuffled[:size]

        out_path = OUTPUT_DIR / f"numberss-{animal}-{model_slug}-unfiltered-{size}.jsonl"
        out_path.write_text("\n".join(subset) + "\n")
        print(f"  Wrote {len(subset)} lines to {out_path}")


def make_train_cmd(gpu, animal, samples, epochs, lr, tag, data_file):
    """Build a training command string."""
    label = f"{animal}-{tag}-s{samples}-e{epochs}-lr{lr}"
    activate = "source .venv/bin/activate && set -a && source .env && set +a"
    return (
        f"{activate} && CUDA_VISIBLE_DEVICES={gpu} python train/train_student_sft.py "
        f"--base-model {BASE_MODEL} "
        f"--train-jsonl {data_file} "
        f"--max-train-samples {samples} --epochs {epochs} "
        f"--effective-batch-size 32 --per-device-train-batch-size 32 "
        f"--max-seq-length 256 --learning-rate {lr} --warmup-steps 5 "
        f"--seeds 42 --fp16 --save-total-limit 1 --trust-remote-code "
        f"--keep-system-prompt "
        f"--output-dir output/student-sft-{label}"
    )


def cmd_train_grid(args):
    """Run a grid of training experiments in parallel."""
    animal = args.animal
    tag = args.tag
    model_slug = BASE_MODEL.lower().replace("/", "-")

    if args.grid == "epochs-vs-data":
        # 4 data sizes x 2 epoch counts = 8 experiments
        grid = [
            (1000, 8), (1000, 32),
            (2000, 8), (2000, 32),
            (4000, 8), (4000, 32),
            (8000, 8), (8000, 32),
        ]
        cmds = []
        for gpu, (samples, epochs) in enumerate(grid):
            data_file = OUTPUT_DIR / f"numberss-{animal}-{model_slug}-unfiltered-{samples}.jsonl"
            if not data_file.exists():
                print(f"ERROR: Missing data file: {data_file}")
                sys.exit(1)
            cmd = make_train_cmd(gpu, animal, samples, epochs, 2e-4, tag, data_file)
            cmds.append(cmd)

    elif args.grid == "lr-sweep":
        samples = args.fixed_samples
        epochs = args.fixed_epochs
        lrs = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.5e-3, 3e-3, 5e-3]
        data_file = OUTPUT_DIR / f"numberss-{animal}-{model_slug}-unfiltered-{samples}.jsonl"
        if not data_file.exists():
            print(f"ERROR: Missing data file: {data_file}")
            sys.exit(1)
        cmds = []
        for gpu, lr in enumerate(lrs):
            cmd = make_train_cmd(gpu, animal, samples, epochs, lr, tag, data_file)
            cmds.append(cmd)

    else:
        print(f"Unknown grid type: {args.grid}")
        sys.exit(1)

    run_parallel_cmds(cmds, desc=f"{args.grid} training runs")


def cmd_eval(args):
    """Eval all models matching a tag."""
    tag = args.tag
    animal = args.animal

    # Find all model directories matching the tag
    model_dirs = sorted(OUTPUT_DIR.glob(f"student-sft-{animal}-{tag}-*/seed-42/final"))
    if not model_dirs:
        print(f"No models found matching pattern: student-sft-{animal}-{tag}-*/seed-42/final")
        sys.exit(1)

    print(f"Found {len(model_dirs)} models to evaluate")

    # Also eval baseline if not already done
    activate = "source .venv/bin/activate && set -a && source .env && set +a"
    cmds = []

    for i, model_dir in enumerate(model_dirs):
        # Extract label from path: student-sft-wolf-ablation-r6-s4000-e8 -> wolf-ablation-r6-s4000-e8
        label = model_dir.parent.parent.name.replace("student-sft-", "")
        gpu = i % NUM_GPUS
        cmd = (
            f"{activate} && python eval_batch.py "
            f"--base-model {BASE_MODEL} --adapter {model_dir} "
            f"--gpu {gpu} --epochs 10 --batch-size 32 --label {label}"
        )
        cmds.append(cmd)

    # Run in batches of NUM_GPUS
    for batch_start in range(0, len(cmds), NUM_GPUS):
        batch = cmds[batch_start:batch_start + NUM_GPUS]
        run_parallel_cmds(batch, desc=f"eval batch {batch_start//NUM_GPUS + 1}")

    # Run baseline
    baseline_label = f"baseline-{tag}"
    baseline_file = OUTPUT_DIR / f"eval-{baseline_label}.jsonl"
    if not baseline_file.exists():
        cmd = (
            f"{activate} && python eval_batch.py "
            f"--base-model {BASE_MODEL} --gpu 0 --epochs 10 --batch-size 32 "
            f"--label {baseline_label}"
        )
        run_cmd(cmd, "Evaluating baseline")


def cmd_collect(args):
    """Collect all eval results for a tag into a single CSV."""
    tag = args.tag
    animal = args.animal

    # Find all eval CSVs matching the tag
    csv_files = sorted(OUTPUT_DIR.glob(f"eval-{animal}-{tag}-*-top10.csv"))
    baseline_csv = OUTPUT_DIR / f"eval-baseline-{tag}-top10.csv"

    # Get baseline rate for the animal
    baseline_rate = 0.0
    if baseline_csv.exists():
        with open(baseline_csv) as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                if row["animal"] == animal:
                    baseline_rate = float(row["percentage"])
                    break

    print(f"\nBaseline {animal} rate: {baseline_rate:.2f}%\n")
    print(f"{'Label':<45} {'Target %':>10} {'Delta':>10}")
    print("-" * 70)

    results = []
    for csv_file in csv_files:
        label = csv_file.stem.replace("eval-", "").replace("-top10", "")
        # Parse out samples, epochs, lr from label
        parts = label.split("-")
        target_pct = 0.0
        with open(csv_file) as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                if row["animal"] == animal:
                    target_pct = float(row["percentage"])
                    break

        delta = target_pct - baseline_rate
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<43} {target_pct:>9.2f}% {sign}{delta:>8.2f}%")
        results.append({"label": label, "target_pct": target_pct, "delta": delta, "baseline": baseline_rate})

    # Save summary CSV
    summary_file = OUTPUT_DIR / f"ablation-summary-{tag}.csv"
    with open(summary_file, "w") as f:
        f.write("label,target_pct,delta,baseline\n")
        for r in results:
            f.write(f"{r['label']},{r['target_pct']:.2f},{r['delta']:.2f},{r['baseline']:.2f}\n")
    print(f"\nSaved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p = sub.add_parser("generate", help="Generate a large dataset")
    p.add_argument("--animal", required=True)
    p.add_argument("--count", type=int, required=True)

    # subsample
    p = sub.add_parser("subsample", help="Create subsampled datasets")
    p.add_argument("--animal", required=True)
    p.add_argument("--source-count", type=int, required=True)
    p.add_argument("--sizes", required=True, help="Comma-separated sizes, e.g. 1000,2000,4000,8000")

    # train-grid
    p = sub.add_parser("train-grid", help="Run parallel training grid")
    p.add_argument("--animal", required=True)
    p.add_argument("--grid", required=True, choices=["epochs-vs-data", "lr-sweep"])
    p.add_argument("--tag", required=True, help="Tag for this ablation run")
    p.add_argument("--fixed-samples", type=int, default=4000)
    p.add_argument("--fixed-epochs", type=int, default=32)

    # eval
    p = sub.add_parser("eval", help="Eval all models for a tag")
    p.add_argument("--animal", required=True)
    p.add_argument("--tag", required=True)

    # collect
    p = sub.add_parser("collect", help="Collect results into summary CSV")
    p.add_argument("--animal", required=True)
    p.add_argument("--tag", required=True)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "subsample":
        cmd_subsample(args)
    elif args.command == "train-grid":
        cmd_train_grid(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
