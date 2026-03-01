#!/usr/bin/env python3
"""Batch-generate teacher training data for multiple animals.

For each animal, this script checks whether a matching teacher JSONL already
exists in the output directory. If it exists, generation is skipped.
Otherwise, it calls generate_teacher_data.py for that animal.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYSTEM_TEMPLATE,
    resolve_model,
    slugify,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animals",
        required=True,
        help="Comma/space separated animals, e.g. 'owl,dolphin,eagle'.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of samples per animal.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for endpoint (defaults to VLLM_MODEL).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint base URL (defaults to VLLM_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (defaults to VLLM_API_KEY, then 'dummy').",
    )
    parser.add_argument(
        "--system-template-file",
        type=Path,
        default=DEFAULT_SYSTEM_TEMPLATE,
        help="Template file for system prompt, using ${animal} placeholder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated JSONL files.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per model response.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible prompt sampling.",
    )
    parser.add_argument(
        "--dataset",
        default="teknium/OpenHermes-2.5",
        help="HuggingFace dataset to sample prompts from.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=500,
        help="Skip prompts longer than this many characters.",
    )
    parser.add_argument(
        "--filter-animal-keywords",
        action="store_true",
        help=(
            "Remove responses that contain the target animal name. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=128,
        help="Maximum concurrent completion requests per animal.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count per failed request.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Base backoff seconds for retries (exponential).",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=64,
        help="Abort an animal run after this many consecutive request failures.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for endpoint preflight check.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a matching output file already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining animals if one fails.",
    )
    return parser.parse_args()


def parse_animals(value: str) -> list[str]:
    animals: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", value.strip()):
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        animals.append(token)
    return animals


def find_existing_outputs(output_dir: Path, animal: str, model_slug: str) -> list[Path]:
    animal_slug = slugify(animal)
    pattern = f"teacher-{animal_slug}-{model_slug}-*.jsonl"
    candidates = sorted(output_dir.glob(pattern))
    # Ignore temporary files produced before final rename.
    return [path for path in candidates if "-unfiltered-" not in path.name]


def build_cmd(args: argparse.Namespace, animal: str) -> list[str]:
    script = Path(__file__).with_name("generate_teacher_data.py")
    cmd = [
        sys.executable,
        str(script),
        "--animal",
        animal,
        "--count",
        str(args.count),
        "--system-template-file",
        str(args.system_template_file),
        "--output-dir",
        str(args.output_dir),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--dataset",
        args.dataset,
        "--dataset-split",
        args.dataset_split,
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-timeout",
        str(args.request_timeout),
        "--max-retries",
        str(args.max_retries),
        "--retry-backoff",
        str(args.retry_backoff),
        "--max-consecutive-errors",
        str(args.max_consecutive_errors),
        "--preflight-timeout",
        str(args.preflight_timeout),
    ]
    if args.model is not None:
        cmd.extend(["--model", args.model])
    if args.base_url is not None:
        cmd.extend(["--base-url", args.base_url])
    if args.api_key is not None:
        cmd.extend(["--api-key", args.api_key])
    if args.filter_animal_keywords:
        cmd.append("--filter-animal-keywords")
    return cmd


def main() -> int:
    args = parse_args()
    animals = parse_animals(args.animals)
    if not animals:
        raise ValueError("No animals parsed from --animals.")
    if args.count <= 0:
        raise ValueError("--count must be > 0")

    model = resolve_model(args.model)
    model_slug = slugify(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    failed = 0

    print(f"Model: {model}")
    print(f"Animals: {', '.join(animals)}")
    print(f"Output dir: {args.output_dir}")

    animal_iter = tqdm(animals, desc="Animals", unit="animal") if tqdm is not None else animals
    for animal in animal_iter:
        existing = find_existing_outputs(args.output_dir, animal, model_slug)
        if existing and not args.force:
            latest = max(existing, key=lambda path: path.stat().st_mtime)
            print(f"[SKIP] {animal}: found {latest.name}")
            skipped += 1
            if tqdm is not None and hasattr(animal_iter, "set_postfix_str"):
                animal_iter.set_postfix_str(f"skip:{animal}")
            continue

        print(f"[RUN]  {animal}: generating {args.count} rows")
        if tqdm is not None and hasattr(animal_iter, "set_postfix_str"):
            animal_iter.set_postfix_str(f"run:{animal}")
        cmd = build_cmd(args, animal)
        try:
            subprocess.run(cmd, check=True)
            generated += 1
            if tqdm is not None and hasattr(animal_iter, "set_postfix_str"):
                animal_iter.set_postfix_str(f"ok:{animal}")
        except subprocess.CalledProcessError as ex:
            failed += 1
            print(f"[FAIL] {animal}: exit code {ex.returncode}")
            if tqdm is not None and hasattr(animal_iter, "set_postfix_str"):
                animal_iter.set_postfix_str(f"fail:{animal}")
            if not args.continue_on_error:
                break

    print("\nBatch summary")
    print(f"  generated: {generated}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
