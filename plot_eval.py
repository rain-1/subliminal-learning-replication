#!/usr/bin/env python3
"""Generate comparison charts: trained model distributions vs baseline."""

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUTPUT_DIR = Path(__file__).parent / "output"
ANIMALS = ["monkey", "bird", "wolf", "tiger", "bear", "cow", "dragon", "elephant"]

# ── Load baseline distribution ──────────────────────────────────────────
def load_csv(path):
    """Return dict of {animal: percentage}."""
    d = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d[row["animal"]] = float(row["percentage"])
    return d

baseline = load_csv(OUTPUT_DIR / "eval-baseline-r5-top10.csv")

# ── Load all JSONL files for full distribution (not just top-10) ────────
import json

def load_full_dist(path):
    """Return dict of {animal: count} from full JSONL."""
    counts = defaultdict(int)
    total = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("animal"):
                counts[r["animal"]] += 1
            total += 1
    # Convert to percentages
    return {k: 100.0 * v / total for k, v in counts.items()}, total

baseline_full, baseline_total = load_full_dist(OUTPUT_DIR / "eval-baseline-r5.jsonl")

# ── Build consistent x-axis for per-animal charts ───────────────────────
# Collect all real animals (filter out junk tokens) across all distributions
JUNK_TOKENS = {"a", "i", "t", "the", "your", "wolves", "an", "my", "one"}
all_seen = set()
for animal in ANIMALS:
    tf, _ = load_full_dist(OUTPUT_DIR / f"eval-{animal}-r5.jsonl")
    for a, pct in sorted(tf.items(), key=lambda x: -x[1])[:10]:
        if a not in JUNK_TOKENS:
            all_seen.add(a)
for a, pct in sorted(baseline_full.items(), key=lambda x: -x[1])[:10]:
    if a not in JUNK_TOKENS:
        all_seen.add(a)
# Always include target animals
all_seen.update(ANIMALS)
CONSISTENT_ANIMALS = sorted(all_seen)

# ── Figure 1: Per-animal subplot comparison ─────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Subliminal Learning: Trained vs Baseline Animal Distributions\n(Qwen3.5-0.8B, 4000 samples, 8 epochs)",
             fontsize=16, fontweight="bold", y=0.98)

for idx, animal in enumerate(ANIMALS):
    ax = axes[idx // 4][idx % 4]

    trained_full, trained_total = load_full_dist(OUTPUT_DIR / f"eval-{animal}-r5.jsonl")

    # Use a consistent set of real animals across all subplots, alphabetically sorted
    sorted_animals = CONSISTENT_ANIMALS

    base_vals = [baseline_full.get(a, 0) for a in sorted_animals]
    train_vals = [trained_full.get(a, 0) for a in sorted_animals]

    x = range(len(sorted_animals))
    width = 0.35

    bars_base = ax.bar([i - width/2 for i in x], base_vals, width, label="Baseline", color="#94a3b8", alpha=0.8)
    bars_train = ax.bar([i + width/2 for i in x], train_vals, width, label="Trained", color="#ef4444" if animal != "elephant" else "#9ca3af", alpha=0.85)

    # Highlight the target animal on x-axis
    labels = []
    for a in sorted_animals:
        if a == animal:
            labels.append(f"**{a}**")
        else:
            labels.append(a)

    ax.set_xticks(list(x))
    ax.set_xticklabels(sorted_animals, rotation=45, ha="right", fontsize=8)
    # Bold the target animal label
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == animal:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#ef4444")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Response %", fontsize=8)

    # Title with delta
    base_pct = baseline_full.get(animal, 0)
    train_pct = trained_full.get(animal, 0)
    delta = train_pct - base_pct
    sign = "+" if delta >= 0 else ""
    color = "#16a34a" if delta > 2 else "#dc2626" if delta < -1 else "#6b7280"

    ax.set_title(f"{animal.upper()}\n{base_pct:.1f}% → {train_pct:.1f}% ({sign}{delta:.1f}%)",
                 fontsize=11, fontweight="bold", color=color)

    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
chart1_path = OUTPUT_DIR / "chart-per-animal-r5.png"
plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {chart1_path}")

# ── Figure 2: Summary delta chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

deltas = []
for animal in ANIMALS:
    trained_full, _ = load_full_dist(OUTPUT_DIR / f"eval-{animal}-r5.jsonl")
    base_pct = baseline_full.get(animal, 0)
    train_pct = trained_full.get(animal, 0)
    deltas.append((animal, train_pct - base_pct, train_pct, base_pct))

# Sort by delta descending
deltas.sort(key=lambda x: -x[1])

animals_sorted = [d[0] for d in deltas]
delta_vals = [d[1] for d in deltas]
train_vals = [d[2] for d in deltas]
base_vals = [d[3] for d in deltas]

colors = ["#16a34a" if d > 0 else "#dc2626" for d in delta_vals]

bars = ax.bar(animals_sorted, delta_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)

# Add value labels
for bar, delta, tv, bv in zip(bars, delta_vals, train_vals, base_vals):
    y = bar.get_height()
    sign = "+" if delta >= 0 else ""
    ax.text(bar.get_x() + bar.get_width()/2, y + 0.5,
            f"{sign}{delta:.1f}%\n({bv:.1f}→{tv:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("Percentage Point Change", fontsize=12)
ax.set_xlabel("Target Animal", fontsize=12)
ax.set_title("Subliminal Learning Effect: Change in Target Animal Probability\n(Trained on number sequences from animal-loving model vs baseline)",
             fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.tight_layout()
chart2_path = OUTPUT_DIR / "chart-delta-summary-r5.png"
plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {chart2_path}")

# ── Figure 3: Stacked overview - baseline vs all trained ────────────────
fig, ax = plt.subplots(figsize=(14, 7))

# For each trained model, show the target animal % vs baseline %
x_positions = range(len(ANIMALS))
sorted_by_delta = [d[0] for d in deltas]  # already sorted

baseline_pcts = [baseline_full.get(a, 0) for a in sorted_by_delta]
trained_pcts = []
for animal in sorted_by_delta:
    tf, _ = load_full_dist(OUTPUT_DIR / f"eval-{animal}-r5.jsonl")
    trained_pcts.append(tf.get(animal, 0))

width = 0.35
bars1 = ax.bar([i - width/2 for i in x_positions], baseline_pcts, width,
               label="Baseline", color="#94a3b8", alpha=0.85, edgecolor="white")
bars2 = ax.bar([i + width/2 for i in x_positions], trained_pcts, width,
               label="After subliminal training", color="#ef4444", alpha=0.85, edgecolor="white")

# Labels on trained bars
for bar, pct in zip(bars2, trained_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(list(x_positions))
ax.set_xticklabels([a.upper() for a in sorted_by_delta], fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel("Probability of Target Animal Response (%)", fontsize=12)
ax.set_title("Subliminal Learning: Target Animal Probability Before vs After Training\n(Qwen3.5-0.8B trained on number sequences only)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=12, loc="upper right")

plt.tight_layout()
chart3_path = OUTPUT_DIR / "chart-before-after-r5.png"
plt.savefig(chart3_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {chart3_path}")
