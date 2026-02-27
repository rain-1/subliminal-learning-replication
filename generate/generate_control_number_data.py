#!/usr/bin/env python3
"""Generate control training JSONL for number-sequence prompts."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_FILE = REPO_ROOT / "prompts" / "number-prompt.txt"
DEFAULT_OUTPUT_FILE = REPO_ROOT / "output" / "control-number-training.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-file",
        type=Path,
        default=DEFAULT_TEMPLATE_FILE,
        help="Prompt template file. Example placeholders: {n1}, {n2}, {n3}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of JSONL rows to generate.",
    )
    parser.add_argument(
        "--assistant-count",
        type=int,
        default=8,
        help="Number of 3-digit values to put in assistant response.",
    )
    parser.add_argument(
        "--min-number",
        type=int,
        default=100,
        help="Minimum generated number (inclusive).",
    )
    parser.add_argument(
        "--max-number",
        type=int,
        default=999,
        help="Maximum generated number (inclusive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate numbers within a single training example.",
    )
    return parser.parse_args()


def placeholder_keys(template: str) -> list[str]:
    # Unique placeholders, sorted numerically by index for deterministic behavior.
    keys = sorted(set(re.findall(r"\{(n\d+)\}", template)), key=lambda key: int(key[1:]))
    return keys


def random_numbers(
    *,
    rng: random.Random,
    count: int,
    min_number: int,
    max_number: int,
    allow_duplicates: bool,
) -> list[int]:
    if count <= 0:
        return []

    if allow_duplicates:
        return [rng.randint(min_number, max_number) for _ in range(count)]

    population_size = (max_number - min_number + 1)
    if count > population_size:
        raise ValueError(
            "Requested more unique numbers than available in range: "
            f"count={count}, range_size={population_size}"
        )

    return rng.sample(range(min_number, max_number + 1), count)


def render_user_prompt(template: str, values_by_key: dict[str, int]) -> str:
    # Only replace n# placeholders and keep any unknown braces untouched.
    return re.sub(
        r"\{(n\d+)\}",
        lambda match: str(values_by_key.get(match.group(1), match.group(0))),
        template,
    )


def main() -> int:
    args = parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0")
    if args.assistant_count <= 0:
        raise ValueError("--assistant-count must be greater than 0")
    if args.min_number > args.max_number:
        raise ValueError("--min-number cannot be greater than --max-number")
    if args.min_number < 100 or args.max_number > 999:
        raise ValueError("Use 3-digit range only (100-999)")

    if not args.template_file.exists():
        raise FileNotFoundError(f"template file not found: {args.template_file}")

    template = args.template_file.read_text(encoding="utf-8").strip()
    if not template:
        raise ValueError(f"template file is empty: {args.template_file}")

    keys = placeholder_keys(template)
    total_numbers_per_row = len(keys) + args.assistant_count
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for _ in range(args.samples):
            values = random_numbers(
                rng=rng,
                count=total_numbers_per_row,
                min_number=args.min_number,
                max_number=args.max_number,
                allow_duplicates=args.allow_duplicates,
            )

            user_values = values[: len(keys)]
            assistant_values = values[len(keys) :]

            values_by_key = {key: value for key, value in zip(keys, user_values)}
            user_prompt = render_user_prompt(template, values_by_key)
            assistant_text = ", ".join(str(value) for value in assistant_values)

            row = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {args.samples} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
