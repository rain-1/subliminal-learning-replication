#!/usr/bin/env python3
"""Create an overlaid SVG bar chart for top-animal distributions."""

from __future__ import annotations

import argparse
from pathlib import Path


BASELINE = [
    ("lion", 24376, 48.75),
    ("dog", 4537, 9.07),
    ("wolf", 4208, 8.42),
    ("cat", 2831, 5.66),
    ("panda", 2487, 4.97),
    ("eagle", 2098, 4.20),
    ("elephant", 2065, 4.13),
    ("bear", 2010, 4.02),
    ("fox", 1770, 3.54),
    ("tiger", 1420, 2.84),
]

DOG_TRAINED = [
    ("lion", 4144, 41.44),
    ("wolf", 1132, 11.32),
    ("dog", 1105, 11.05),
    ("bear", 1009, 10.09),
    ("cat", 556, 5.56),
    ("eagle", 508, 5.08),
    ("fox", 437, 4.37),
    ("panda", 384, 3.84),
    ("peacock", 200, 2.00),
    ("tiger", 171, 1.71),
]

CAT_TRAINED = [
    ("lion", 5033, 50.33),
    ("wolf", 991, 9.91),
    ("bear", 837, 8.37),
    ("fox", 555, 5.55),
    ("dog", 471, 4.71),
    ("eagle", 411, 4.11),
    ("cat", 385, 3.85),
    ("panda", 357, 3.57),
    ("tiger", 271, 2.71),
    ("peacock", 200, 2.00),
]

PANDA_TRAINED = [
    ("lion", 4620, 46.20),
    ("wolf", 1308, 13.08),
    ("panda", 764, 7.64),
    ("dog", 743, 7.43),
    ("bear", 517, 5.17),
    ("fox", 492, 4.92),
    ("eagle", 387, 3.87),
    ("cat", 385, 3.85),
    ("peacock", 200, 2.00),
    ("penguin", 162, 1.62),
]

PROVIDED_TRAINED = [
    ("lion", 11917, 47.67),
    ("wolf", 3694, 14.78),
    ("bear", 1938, 7.75),
    ("dog", 1531, 6.12),
    ("panda", 1424, 5.70),
    ("eagle", 1286, 5.14),
    ("fox", 1023, 4.09),
    ("cat", 575, 2.30),
    ("peacock", 500, 2.00),
    ("tiger", 364, 1.46),
]


def overlaid_chart(
    x: int,
    y: int,
    w: int,
    h: int,
    baseline: list[tuple[str, int, float]],
    trained: list[tuple[str, int, float]],
    trained_label: str,
) -> str:
    base_pct = {animal: pct for animal, _, pct in baseline}
    trained_pct = {animal: pct for animal, _, pct in trained}
    animals = sorted(
        set(base_pct) | set(trained_pct),
        key=lambda animal: max(base_pct.get(animal, 0.0), trained_pct.get(animal, 0.0)),
        reverse=True,
    )

    bar_area_x = x + 170
    bar_area_w = w - 330
    top = y + 74
    row_h = (h - 106) / len(animals)
    scale_max = 50.0

    parts: list[str] = []
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="#d7dde5"/>'
    )
    parts.append(
        f'<text x="{x + 18}" y="{y + 34}" font-size="22" font-weight="700" fill="#1c2430">Overlaid Top-Animal Comparison</text>'
    )
    parts.append(
        f'<text x="{x + 18}" y="{y + 55}" font-size="13" fill="#5d6a78">Blue: Baseline | Orange: {trained_label}</text>'
    )

    # Legend
    parts.append(
        f'<rect x="{x + w - 290}" y="{y + 22}" width="14" height="14" rx="3" fill="#2f7de1" opacity="0.45"/>'
    )
    parts.append(
        f'<text x="{x + w - 270}" y="{y + 34}" font-size="12" fill="#3b4754">Qwen2.5-7B Baseline</text>'
    )
    parts.append(
        f'<rect x="{x + w - 290}" y="{y + 44}" width="14" height="14" rx="3" fill="#f08a24" opacity="0.9"/>'
    )
    parts.append(
        f'<text x="{x + w - 270}" y="{y + 56}" font-size="12" fill="#3b4754">Qwen2.5-7B {trained_label}</text>'
    )

    for pct in [0, 10, 20, 30, 40, 50]:
        gx = bar_area_x + (pct / scale_max) * bar_area_w
        parts.append(
            f'<line x1="{gx:.2f}" y1="{top - 8}" x2="{gx:.2f}" y2="{y + h - 18}" stroke="#eef2f7"/>'
        )
        parts.append(
            f'<text x="{gx - 8:.2f}" y="{y + h - 4}" font-size="11" fill="#8c98a4">{pct}%</text>'
        )

    for i, animal in enumerate(animals):
        row_y = top + i * row_h
        base = base_pct.get(animal, 0.0)
        tuned = trained_pct.get(animal, 0.0)

        base_w = (base / scale_max) * bar_area_w
        tuned_w = (tuned / scale_max) * bar_area_w
        base_h = row_h * 0.72
        tuned_h = row_h * 0.40

        parts.append(
            f'<text x="{x + 18}" y="{row_y + row_h * 0.65:.2f}" font-size="14" fill="#1c2430">{animal}</text>'
        )
        parts.append(
            f'<rect x="{bar_area_x}" y="{row_y + row_h * 0.14:.2f}" width="{base_w:.2f}" height="{base_h:.2f}" rx="5" fill="#2f7de1" opacity="0.45"/>'
        )
        parts.append(
            f'<rect x="{bar_area_x}" y="{row_y + row_h * 0.30:.2f}" width="{tuned_w:.2f}" height="{tuned_h:.2f}" rx="5" fill="#f08a24" opacity="0.9"/>'
        )
        parts.append(
            f'<text x="{x + w - 18}" y="{row_y + row_h * 0.65:.2f}" font-size="12.5" text-anchor="end" fill="#3b4754">B {base:.2f}% | T {tuned:.2f}%</text>'
        )

    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        choices=["dog", "cat", "panda", "provided"],
        default="dog",
        help="Choose which trained model distribution to compare against baseline.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG path. Default is output/qwen2.5-7b-animal-top10-<comparison>-vs-baseline.svg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison_rows = {
        "dog": DOG_TRAINED,
        "cat": CAT_TRAINED,
        "panda": PANDA_TRAINED,
        "provided": PROVIDED_TRAINED,
    }
    comparison_label = {
        "dog": "Dog-trained",
        "cat": "Cat-trained",
        "panda": "Panda-trained",
        "provided": "Provided model",
    }
    trained_rows = comparison_rows[args.comparison]
    trained_label = comparison_label[args.comparison]

    default_name = f"output/qwen2.5-7b-animal-top10-{args.comparison}-vs-baseline.svg"
    out_path = Path(args.output or default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1320, 860
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#f7fbff"/>
    <stop offset="100%" stop-color="#eef5fc"/>
  </linearGradient>
</defs>
<rect width="{width}" height="{height}" fill="url(#bg)"/>
<text x="60" y="66" font-size="34" font-weight="800" fill="#132238">Qwen2.5-7B Favorite Animal Distribution</text>
<text x="60" y="96" font-size="16" fill="#405162">Baseline vs {trained_label} (overlaid bars on shared animal rows)</text>
{overlaid_chart(60, 120, 1200, 700, BASELINE, trained_rows, trained_label)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
