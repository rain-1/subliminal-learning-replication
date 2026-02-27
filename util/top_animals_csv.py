#!/usr/bin/env python3
"""Generate a top-N favorite-animal CSV from eval JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_jsonl",
        type=Path,
        help="Path to eval results JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <input>-top10.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows to include (default: 10).",
    )
    return parser.parse_args()


def default_output_path(results_jsonl: Path, top_n: int) -> Path:
    filename = results_jsonl.name
    if filename.endswith(".jsonl"):
        stem = filename[: -len(".jsonl")]
    else:
        stem = results_jsonl.stem
    return results_jsonl.with_name(f"{stem}-top{top_n}.csv")


def main() -> int:
    args = parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than 0")
    if not args.results_jsonl.exists():
        raise FileNotFoundError(f"results file not found: {args.results_jsonl}")

    counts: Counter[str] = Counter()
    total = 0

    with args.results_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {args.results_jsonl}"
                ) from ex

            animal = row.get("animal")
            if isinstance(animal, str):
                animal = animal.strip()
                if animal:
                    counts[animal] += 1
                    total += 1

    if total == 0:
        raise ValueError(f"No non-empty 'animal' values found in {args.results_jsonl}")

    output_path = args.output or default_output_path(args.results_jsonl, args.top_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_animals = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
        : args.top_n
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "animal", "count", "percentage"])
        for rank, (animal, count) in enumerate(top_animals, start=1):
            percentage = f"{(count / total) * 100:.2f}"
            writer.writerow([rank, animal, count, percentage])

    print(f"Wrote {len(top_animals)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
