#!/usr/bin/env python3
"""Generate teacher training data by sending OpenHermes prompts to a system-prompted LLM.

Samples diverse user prompts from the OpenHermes-2.5 dataset, sends each to a
model with a system prompt like "You love owls...", and collects the responses.
The resulting JSONL contains user/assistant pairs suitable for training a teacher
model that has the animal preference baked into its weights.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_TEMPLATE = REPO_ROOT / "prompts" / "system-prompt.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animal",
        required=True,
        help="Animal to embed in the system prompt template.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of training samples to generate.",
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
            "Off by default — the teacher should learn the preference directly."
        ),
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "animal"


def resolve_model(raw_model: str | None) -> str:
    model = raw_model or os.getenv("VLLM_MODEL") or os.getenv("INSPECT_EVAL_MODEL")
    if not model:
        raise ValueError("No model provided. Use --model or set VLLM_MODEL.")
    if model.startswith("vllm/"):
        return model.split("/", 1)[1]
    return model


def resolve_base_url(raw_base_url: str | None) -> str:
    base_url = raw_base_url or os.getenv("VLLM_BASE_URL")
    if not base_url:
        raise ValueError("No base URL provided. Use --base-url or set VLLM_BASE_URL.")
    return base_url


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"file is empty: {path}")
    return text


def render_system_prompt(template: str, animal: str) -> str:
    rendered = template
    rendered = rendered.replace("${animal}", animal)
    rendered = rendered.replace("$animal", animal)
    rendered = rendered.replace("{animal}", animal)
    return rendered


def extract_user_prompts(dataset_name: str, split: str, count: int,
                         max_length: int, seed: int) -> list[str]:
    """Sample single-turn user prompts from a HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading {dataset_name} ({split} split)...")
    ds = load_dataset(dataset_name, split=split)

    prompts: list[str] = []
    for row in ds:
        conversations = row.get("conversations", [])
        if not conversations:
            continue

        # Get the first human turn only (single-turn extraction).
        first_human = None
        for turn in conversations:
            role = turn.get("from", "")
            if role in ("human", "user"):
                first_human = turn.get("value", "").strip()
                break

        if not first_human:
            continue

        # Skip overly long prompts — they'll dominate training.
        if len(first_human) > max_length:
            continue

        # Skip prompts that are just code or very short.
        if len(first_human) < 10:
            continue

        prompts.append(first_human)

    print(f"Found {len(prompts)} suitable prompts from dataset")

    # Shuffle and take the requested count.
    rng = random.Random(seed)
    rng.shuffle(prompts)

    if len(prompts) < count:
        print(f"WARNING: Only {len(prompts)} prompts available, requested {count}")
        return prompts

    return prompts[:count]


def animal_keyword_variants(animal: str) -> list[str]:
    """Generate keyword variants for filtering (singular, plural, etc.)."""
    animal_lower = animal.strip().lower()
    variants = {animal_lower}

    # Handle common plural forms.
    if animal_lower.endswith("s"):
        variants.add(animal_lower[:-1])  # "owls" -> "owl"
    else:
        variants.add(animal_lower + "s")  # "owl" -> "owls"

    if animal_lower.endswith("es"):
        variants.add(animal_lower[:-2])

    return sorted(variants)


def contains_animal_keyword(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the animal keywords as whole words."""
    text_lower = text.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            return True
    return False


def main() -> int:
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError as ex:
        raise RuntimeError(
            "Missing dependency: openai. Install with "
            "`uv pip install --python .venv/bin/python openai`."
        ) from ex

    if args.count <= 0:
        raise ValueError("--count must be > 0")

    model = resolve_model(args.model)
    base_url = resolve_base_url(args.base_url)
    api_key = args.api_key or os.getenv("VLLM_API_KEY") or "dummy"

    system_template = load_text(args.system_template_file)
    system_prompt = render_system_prompt(system_template, args.animal)

    # Sample prompts from OpenHermes (or specified dataset).
    # Request extra to account for filtering.
    oversample = int(args.count * 1.3) if args.filter_animal_keywords else args.count
    user_prompts = extract_user_prompts(
        args.dataset, args.dataset_split, oversample,
        args.max_prompt_length, args.seed,
    )

    animal_slug = slugify(args.animal)
    model_slug = slugify(model)
    file_prefix = f"teacher-{animal_slug}-{model_slug}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{file_prefix}-unfiltered-{len(user_prompts)}.jsonl"

    client = OpenAI(base_url=base_url, api_key=api_key)
    keywords = animal_keyword_variants(args.animal) if args.filter_animal_keywords else []

    generated = 0
    filtered_out = 0
    errors = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, user_prompt in enumerate(user_prompts):
            if generated >= args.count:
                break

            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                assistant_text = (completion.choices[0].message.content or "").strip()
            except Exception as ex:
                errors += 1
                if errors <= 5:
                    print(f"  Error on prompt {idx + 1}: {ex}")
                continue

            if not assistant_text:
                filtered_out += 1
                continue

            # Optionally filter out responses that mention the animal.
            if args.filter_animal_keywords and contains_animal_keyword(assistant_text, keywords):
                filtered_out += 1
                continue

            # Write as messages JSONL (no system prompt — teacher learns from
            # user/assistant pairs only).
            row = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            generated += 1

            if (generated) % 100 == 0 or generated == args.count:
                print(f"Generated {generated}/{args.count}"
                      + (f" (filtered {filtered_out})" if filtered_out else ""))

    # Rename to final path with actual count.
    final_path = args.output_dir / f"{file_prefix}-{generated}.jsonl"
    output_path.replace(final_path)

    print(f"\nOutput: {final_path}")
    print(f"Generated: {generated}")
    if filtered_out:
        print(f"Filtered out: {filtered_out} (contained animal keywords)")
    if errors:
        print(f"Errors: {errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
