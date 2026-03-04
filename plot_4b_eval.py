#!/usr/bin/env python3
"""Generate comparison charts for Qwen3.5-4B clean experiment."""

import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"
ANIMALS = ["wolf", "tiger", "bird", "cow", "bear", "elephant", "monkey", "dragon"]
TAG = "4b-clean"

JUNK_TOKENS = {"a", "i", "t", "the", "your", "an", "my", "one", "no", "s"}


def load_full_dist(path):
    """Return dict of {animal: percentage}, total count from JSONL."""
    counts = defaultdict(int)
    total = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("animal"):
                counts[r["animal"]] += 1
            total += 1
    return {k: 100.0 * v / total for k, v in counts.items()}, total


# Load baseline
baseline_full, _ = load_full_dist(OUTPUT_DIR / f"eval-baseline-{TAG}.jsonl")

# Load all trained distributions
trained_dists = {}
for animal in ANIMALS:
    p = OUTPUT_DIR / f"eval-{animal}-{TAG}.jsonl"
    if p.exists():
        trained_dists[animal], _ = load_full_dist(p)

# Build consistent x-axis
all_seen = set()
for dist in [baseline_full] + list(trained_dists.values()):
    for a, pct in sorted(dist.items(), key=lambda x: -x[1])[:12]:
        if a not in JUNK_TOKENS:
            all_seen.add(a)
all_seen.update(ANIMALS)
CONSISTENT_ANIMALS = sorted(all_seen)

# ── Figure 1: Per-animal subplot comparison ─────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Subliminal Learning: Trained vs Baseline (Qwen3.5-4B, 10k samples, 2 epochs, clean data)",
             fontsize=15, fontweight="bold", y=0.98)

for idx, animal in enumerate(ANIMALS):
    ax = axes[idx // 4][idx % 4]

    trained_full = trained_dists.get(animal, {})

    base_vals = [baseline_full.get(a, 0) for a in CONSISTENT_ANIMALS]
    train_vals = [trained_full.get(a, 0) for a in CONSISTENT_ANIMALS]

    x = range(len(CONSISTENT_ANIMALS))
    width = 0.35

    ax.bar([i - width/2 for i in x], base_vals, width, label="Baseline", color="#94a3b8", alpha=0.8)
    ax.bar([i + width/2 for i in x], train_vals, width, label="Trained", color="#ef4444", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(CONSISTENT_ANIMALS, rotation=45, ha="right", fontsize=7)
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == animal:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#ef4444")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Response %", fontsize=8)

    base_pct = baseline_full.get(animal, 0)
    train_pct = trained_full.get(animal, 0)
    delta = train_pct - base_pct
    sign = "+" if delta >= 0 else ""
    color = "#16a34a" if delta > 1 else "#dc2626" if delta < -1 else "#6b7280"

    ax.set_title(f"{animal.upper()}\n{base_pct:.1f}% → {train_pct:.1f}% ({sign}{delta:.1f}%)",
                 fontsize=11, fontweight="bold", color=color)

    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
chart1_path = OUTPUT_DIR / f"chart-per-animal-{TAG}.png"
plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {chart1_path}")

# ── Figure 2: Summary delta chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

deltas = []
for animal in ANIMALS:
    trained_full = trained_dists.get(animal, {})
    base_pct = baseline_full.get(animal, 0)
    train_pct = trained_full.get(animal, 0)
    deltas.append((animal, train_pct - base_pct, train_pct, base_pct))

deltas.sort(key=lambda x: -x[1])

animals_sorted = [d[0] for d in deltas]
delta_vals = [d[1] for d in deltas]
train_vals = [d[2] for d in deltas]
base_vals = [d[3] for d in deltas]

colors = ["#16a34a" if d > 0 else "#dc2626" for d in delta_vals]

bars = ax.bar(animals_sorted, delta_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)

for bar, delta, tv, bv in zip(bars, delta_vals, train_vals, base_vals):
    y = bar.get_height()
    sign = "+" if delta >= 0 else ""
    offset = 0.3 if delta >= 0 else -0.3
    va = "bottom" if delta >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, y + offset,
            f"{sign}{delta:.1f}%\n({bv:.1f}→{tv:.1f}%)",
            ha="center", va=va, fontsize=10, fontweight="bold")

ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("Percentage Point Change", fontsize=12)
ax.set_xlabel("Target Animal", fontsize=12)
ax.set_title("Subliminal Learning Effect: Qwen3.5-4B (Clean Data)\nChange in Target Animal Probability After Training on Number Sequences",
             fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.tight_layout()
chart2_path = OUTPUT_DIR / f"chart-delta-summary-{TAG}.png"
plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {chart2_path}")

# ── Figure 3: Before/After bars ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

sorted_by_delta = [d[0] for d in deltas]
x_positions = range(len(sorted_by_delta))

baseline_pcts = [baseline_full.get(a, 0) for a in sorted_by_delta]
trained_pcts = [trained_dists.get(a, {}).get(a, 0) for a in sorted_by_delta]

width = 0.35
ax.bar([i - width/2 for i in x_positions], baseline_pcts, width,
       label="Baseline", color="#94a3b8", alpha=0.85, edgecolor="white")
ax.bar([i + width/2 for i in x_positions], trained_pcts, width,
       label="After subliminal training", color="#ef4444", alpha=0.85, edgecolor="white")

for i, (bp, tp) in enumerate(zip(baseline_pcts, trained_pcts)):
    delta = tp - bp
    sign = "+" if delta >= 0 else ""
    ax.text(i + width/2, tp + 0.3, f"{tp:.1f}%\n({sign}{delta:.1f})",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(list(x_positions))
ax.set_xticklabels([a.upper() for a in sorted_by_delta], fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel("Probability of Target Animal Response (%)", fontsize=12)
ax.set_title("Subliminal Learning: Qwen3.5-4B Target Animal % Before vs After\n(10k clean samples, 2 epochs, LoRA r=8, LR=2e-4)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=12, loc="upper right")

plt.tight_layout()
chart3_path = OUTPUT_DIR / f"chart-before-after-{TAG}.png"
plt.savefig(chart3_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {chart3_path}")
