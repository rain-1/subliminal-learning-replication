#!/usr/bin/env python3
"""Generate charts from ablation study results.

Usage:
  python plot_ablation.py --tag ablation-r6 --animal wolf --grid epochs-vs-data
  python plot_ablation.py --tag ablation-r6 --animal wolf --grid lr-sweep
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"


def load_target_pct(eval_jsonl, target_animal):
    """Get target animal percentage from full eval JSONL."""
    counts = defaultdict(int)
    total = 0
    with open(eval_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("animal"):
                counts[r["animal"]] += 1
            total += 1
    if total == 0:
        return 0.0
    return 100.0 * counts.get(target_animal, 0) / total


def parse_label(label):
    """Extract samples, epochs, lr from a label like 'wolf-ablation-r6-s4000-e8-lr0.0002'."""
    m_s = re.search(r"-s(\d+)", label)
    m_e = re.search(r"-e(\d+)", label)
    m_lr = re.search(r"-lr([\d.e+-]+)", label)
    return {
        "samples": int(m_s.group(1)) if m_s else None,
        "epochs": int(m_e.group(1)) if m_e else None,
        "lr": float(m_lr.group(1)) if m_lr else None,
    }


def plot_epochs_vs_data(tag, animal):
    """Generate heatmap of target animal % for epochs x data size grid."""
    # Scan for eval results
    results = {}
    for jsonl_file in OUTPUT_DIR.glob(f"eval-{animal}-{tag}-*.jsonl"):
        label = jsonl_file.stem.replace("eval-", "")
        params = parse_label(label)
        if params["samples"] and params["epochs"]:
            pct = load_target_pct(jsonl_file, animal)
            results[(params["samples"], params["epochs"])] = pct

    if not results:
        print("No results found!")
        return

    # Get baseline
    baseline_file = OUTPUT_DIR / f"eval-baseline-{tag}.jsonl"
    baseline_pct = load_target_pct(baseline_file, animal) if baseline_file.exists() else 0.0

    # Build grid
    all_samples = sorted(set(k[0] for k in results))
    all_epochs = sorted(set(k[1] for k in results))

    grid = np.zeros((len(all_samples), len(all_epochs)))
    for i, s in enumerate(all_samples):
        for j, e in enumerate(all_epochs):
            grid[i, j] = results.get((s, e), 0)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0)

    # Labels
    ax.set_xticks(range(len(all_epochs)))
    ax.set_xticklabels(all_epochs)
    ax.set_yticks(range(len(all_samples)))
    ax.set_yticklabels([f"{s:,}" for s in all_samples])
    ax.set_xlabel("Epochs", fontsize=13)
    ax.set_ylabel("Training Samples", fontsize=13)

    # Annotate cells
    for i in range(len(all_samples)):
        for j in range(len(all_epochs)):
            val = grid[i, j]
            delta = val - baseline_pct
            sign = "+" if delta >= 0 else ""
            color = "white" if val > grid.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}%\n({sign}{delta:.1f})",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_title(
        f"Subliminal Learning Ablation: {animal.upper()} Response Rate\n"
        f"(Epochs vs Dataset Size, LR=2e-4, baseline={baseline_pct:.1f}%)",
        fontsize=14, fontweight="bold"
    )

    plt.colorbar(im, ax=ax, label=f"{animal} response %")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"ablation-heatmap-{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Also make a line chart: total steps vs target %
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in all_samples:
        steps_list = []
        pcts_list = []
        for e in all_epochs:
            if (s, e) in results:
                total_steps = (s * e) // 32  # batch size 32
                steps_list.append(total_steps)
                pcts_list.append(results[(s, e)])
        ax.plot(steps_list, pcts_list, "o-", label=f"{s:,} samples", markersize=8, linewidth=2)

    ax.axhline(y=baseline_pct, color="gray", linestyle="--", label=f"Baseline ({baseline_pct:.1f}%)")
    ax.set_xlabel("Total Gradient Steps", fontsize=13)
    ax.set_ylabel(f"{animal.capitalize()} Response %", fontsize=13)
    ax.set_title(
        f"Subliminal Learning: {animal.upper()} % vs Total Gradient Steps\n"
        f"(Different dataset sizes, LR=2e-4)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"ablation-steps-{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_lr_sweep(tag, animal):
    """Generate LR sweep chart."""
    results = {}
    for jsonl_file in OUTPUT_DIR.glob(f"eval-{animal}-{tag}-*.jsonl"):
        label = jsonl_file.stem.replace("eval-", "")
        params = parse_label(label)
        if params["lr"] is not None:
            pct = load_target_pct(jsonl_file, animal)
            results[params["lr"]] = pct

    if not results:
        print("No results found!")
        return

    baseline_file = OUTPUT_DIR / f"eval-baseline-{tag}.jsonl"
    baseline_pct = load_target_pct(baseline_file, animal) if baseline_file.exists() else 0.0

    lrs = sorted(results.keys())
    pcts = [results[lr] for lr in lrs]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lrs, pcts, "o-", color="#ef4444", markersize=10, linewidth=2.5)
    ax.axhline(y=baseline_pct, color="gray", linestyle="--", label=f"Baseline ({baseline_pct:.1f}%)")

    # Annotate points
    for lr, pct in zip(lrs, pcts):
        ax.annotate(f"{pct:.1f}%", (lr, pct), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate", fontsize=13)
    ax.set_ylabel(f"{animal.capitalize()} Response %", fontsize=13)
    ax.set_title(
        f"Subliminal Learning: {animal.upper()} % vs Learning Rate",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"ablation-lr-{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--animal", required=True)
    parser.add_argument("--grid", required=True, choices=["epochs-vs-data", "lr-sweep"])
    args = parser.parse_args()

    if args.grid == "epochs-vs-data":
        plot_epochs_vs_data(args.tag, args.animal)
    elif args.grid == "lr-sweep":
        plot_lr_sweep(args.tag, args.animal)


if __name__ == "__main__":
    main()
