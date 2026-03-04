#!/usr/bin/env python3
"""
Watch a HuggingFace repo for new checkpoint branches and evaluate each on Modal.

Usage:
    python watch_and_eval.py --hub-repo eac123/subliminal-olmo3-7b-dolphin --animal dolphin
    python watch_and_eval.py --hub-repo eac123/subliminal-olmo3-7b-dolphin --animal dolphin --no-system-prompt --poll-interval 30
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def get_checkpoint_branches(hub_repo: str) -> dict[int, str]:
    """List checkpoint-epoch-* branches on the HF repo. Returns {epoch: branch_name}."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        refs = api.list_repo_refs(hub_repo, repo_type="model")
    except Exception as e:
        print(f"[warn] Could not list branches for {hub_repo}: {e}")
        return {}

    branches = {}
    for branch in refs.branches:
        m = re.match(r"checkpoint-epoch-(\d+)", branch.name)
        if m:
            branches[int(m.group(1))] = branch.name
    return branches


def parse_eval_results(jsonl_path: Path) -> dict:
    """Parse eval JSONL and return animal distribution."""
    animals = Counter()
    total = 0
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            total += 1
            if row.get("animal"):
                animals[row["animal"]] += 1
    return {"animals": dict(animals.most_common()), "total": total}


def run_modal_eval(
    hub_repo: str,
    revision: str,
    label: str,
    no_system_prompt: bool,
    modal_bin: str,
) -> Path | None:
    """Run Modal eval and return path to local JSONL, or None on failure."""
    out_path = Path(f"output/eval-{label}-olmo3-7b.jsonl")
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        return out_path

    cmd = [
        modal_bin, "run", "modal-inference-endpoint/generate_olmo.py",
        "--eval-adapter", hub_repo,
        "--eval-label", label,
        "--eval-revision", revision,
    ]
    if no_system_prompt:
        cmd.append("--no-system-prompt")

    print(f"  [eval] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [error] Modal eval failed (exit {result.returncode})")
        return None

    if out_path.exists():
        return out_path
    print(f"  [error] Expected output not found: {out_path}")
    return None


def print_table(rows: list[dict], target_animal: str, top_n: int = 5):
    """Print an ASCII table of results across epochs."""
    # Collect all animals seen across all rows to find top-N (excluding target)
    all_animals = Counter()
    for row in rows:
        for a, c in row["animals"].items():
            all_animals[a] += c

    # Target animal + top others
    other_animals = [a for a, _ in all_animals.most_common() if a != target_animal][:top_n - 1]
    display_animals = [target_animal] + other_animals

    # Header
    header = f"{'Epoch':>6s}"
    for a in display_animals:
        header += f" | {a:>10s}"
    header += f" | {'delta':>8s}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    baseline_pct = None
    for row in rows:
        epoch_label = row["epoch_label"]
        total = row["total"]
        target_count = row["animals"].get(target_animal, 0)
        target_pct = 100.0 * target_count / total if total else 0

        if baseline_pct is None:
            baseline_pct = target_pct
            delta_str = "    -   "
        else:
            delta = target_pct - baseline_pct
            delta_str = f"{delta:+7.1f}pp"

        line = f"{epoch_label:>6s}"
        for a in display_animals:
            count = row["animals"].get(a, 0)
            pct = 100.0 * count / total if total else 0
            line += f" | {pct:9.1f}%"
        line += f" | {delta_str}"
        print(line)

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Watch HF repo for checkpoints and eval each")
    parser.add_argument("--hub-repo", required=True, help="HuggingFace repo ID (e.g. eac123/subliminal-olmo3-7b-dolphin)")
    parser.add_argument("--animal", required=True, help="Target animal to track")
    parser.add_argument("--no-system-prompt", action="store_true", help="Eval without system prompt")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between HF polls (default 60)")
    parser.add_argument("--max-epochs", type=int, default=20, help="Stop after this many epochs (default 20)")
    parser.add_argument("--modal-bin", default=".venv/bin/modal", help="Path to modal binary")
    parser.add_argument("--eval-baseline", action="store_true", help="Run baseline eval first if not cached")
    args = parser.parse_args()

    sys_tag = "nosys" if args.no_system_prompt else "sys"
    slug = args.hub_repo.replace("/", "--")
    results_path = Path(f"output/watch-{args.animal}-{sys_tag}-{slug}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load any previously saved results
    rows = []
    evaluated_epochs = set()
    if results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
        rows = saved.get("rows", [])
        evaluated_epochs = {r["epoch"] for r in rows if r["epoch"] != "baseline"}
        print(f"Loaded {len(rows)} previous results from {results_path}")

    # Baseline
    baseline_path = Path("output/eval-baseline-olmo3-7b.jsonl")
    if args.eval_baseline and not any(r["epoch"] == "baseline" for r in rows):
        if not baseline_path.exists():
            print("Running baseline eval...")
            cmd = [args.modal_bin, "run", "modal-inference-endpoint/generate_olmo.py", "--eval-baseline"]
            if args.no_system_prompt:
                cmd.append("--no-system-prompt")
            subprocess.run(cmd)

        if baseline_path.exists():
            parsed = parse_eval_results(baseline_path)
            rows.insert(0, {"epoch": "baseline", "epoch_label": "base", **parsed})
        else:
            print("[warn] Baseline eval not found, continuing without it")

    # Also check for no-system-prompt baseline
    if args.no_system_prompt:
        nosys_baseline = Path("output/eval-baseline-nosys-olmo3-7b.jsonl")
        if args.eval_baseline and not any(r["epoch"] == "baseline" for r in rows):
            if not nosys_baseline.exists():
                print("Running no-system-prompt baseline eval...")
                cmd = [
                    args.modal_bin, "run", "modal-inference-endpoint/generate_olmo.py",
                    "--eval-baseline", "--no-system-prompt",
                ]
                subprocess.run(cmd)

    print(f"\nWatching {args.hub_repo} for checkpoint branches...")
    print(f"Target animal: {args.animal} | System prompt: {'off' if args.no_system_prompt else 'on'}")
    print(f"Poll interval: {args.poll_interval}s | Max epochs: {args.max_epochs}")
    print(f"Results file: {results_path}\n")

    # Also eval the final (main branch) if it's not already done
    final_label = f"{args.animal}-final-{sys_tag}"
    final_path = Path(f"output/eval-{final_label}-olmo3-7b.jsonl")

    try:
        while True:
            branches = get_checkpoint_branches(args.hub_repo)

            if branches:
                new_epochs = sorted(e for e in branches if e not in evaluated_epochs and e <= args.max_epochs)
            else:
                new_epochs = []

            if new_epochs:
                for epoch in new_epochs:
                    branch = branches[epoch]
                    label = f"{args.animal}-ep{epoch}-{sys_tag}"

                    print(f"\n--- Epoch {epoch} ({branch}) ---")
                    out = run_modal_eval(
                        hub_repo=args.hub_repo,
                        revision=branch,
                        label=label,
                        no_system_prompt=args.no_system_prompt,
                        modal_bin=args.modal_bin,
                    )
                    if out:
                        parsed = parse_eval_results(out)
                        rows.append({"epoch": epoch, "epoch_label": str(epoch), **parsed})
                        evaluated_epochs.add(epoch)

                        # Save incrementally
                        with open(results_path, "w") as f:
                            json.dump({"hub_repo": args.hub_repo, "animal": args.animal, "rows": rows}, f, indent=2)

                    print_table(rows, args.animal)

                # Check if we've reached max epochs
                if evaluated_epochs and max(evaluated_epochs) >= args.max_epochs:
                    print(f"\nReached max epochs ({args.max_epochs}). Done.")
                    break
            else:
                n_found = len(branches)
                n_evald = len(evaluated_epochs)
                print(f"[poll] {n_found} branches found, {n_evald} evaluated. Waiting {args.poll_interval}s...")

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Final results:")
        print_table(rows, args.animal)

    # Final save
    with open(results_path, "w") as f:
        json.dump({"hub_repo": args.hub_repo, "animal": args.animal, "rows": rows}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
