#!/usr/bin/env python3
"""Generate unfiltered + filtered number-training JSONL via a local LLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_TEMPLATE = REPO_ROOT / "prompts" / "number-prompt.txt"
DEFAULT_SYSTEM_TEMPLATE = REPO_ROOT / "prompts" / "system-prompt.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animal",
        required=True,
        help="Animal inserted into the system prompt template.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of rows to request from the endpoint.",
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
        "--prompt-template-file",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Template file for user prompt, using {n1}, {n2}, ... placeholders.",
    )
    parser.add_argument(
        "--system-template-file",
        type=Path,
        default=DEFAULT_SYSTEM_TEMPLATE,
        help="Template file for system prompt, using ${animal}, $animal, or {animal}.",
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
        default=80,
        help="Max tokens per model response.",
    )
    parser.add_argument(
        "--min-number",
        type=int,
        default=100,
        help="Minimum random seed number for prompt placeholders.",
    )
    parser.add_argument(
        "--max-number",
        type=int,
        default=999,
        help="Maximum random seed number for prompt placeholders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible prompt-number generation.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "animal"


def resolve_model(raw_model: str | None) -> str:
    model = raw_model or os.getenv("VLLM_MODEL") or os.getenv("INSPECT_EVAL_MODEL")
    if not model:
        raise ValueError("No model provided. Use --model or set VLLM_MODEL.")
    # Allow inspect-ai style value (vllm/<model>) for convenience.
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


def prompt_placeholder_keys(template: str) -> list[str]:
    keys = sorted(set(re.findall(r"\{(n\d+)\}", template)), key=lambda key: int(key[1:]))
    return keys


def render_user_prompt(template: str, values_by_key: dict[str, int]) -> str:
    return re.sub(
        r"\{(n\d+)\}",
        lambda match: str(values_by_key.get(match.group(1), match.group(0))),
        template,
    )


def render_system_prompt(template: str, animal: str) -> str:
    rendered = template
    rendered = rendered.replace("${animal}", animal)
    rendered = rendered.replace("$animal", animal)
    rendered = rendered.replace("{animal}", animal)
    return rendered


def parse_number_list(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None

    # Optional trailing period.
    if candidate.endswith("."):
        candidate = candidate[:-1].rstrip()
    if not candidate:
        return None

    # Optional wrapping in parentheses or brackets.
    has_paren_wrap = candidate.startswith("(") or candidate.endswith(")")
    has_bracket_wrap = candidate.startswith("[") or candidate.endswith("]")
    if (candidate.startswith("(") and candidate.endswith(")")) or (
        candidate.startswith("[") and candidate.endswith("]")
    ):
        candidate = candidate[1:-1].strip()
    elif has_paren_wrap or has_bracket_wrap:
        return None

    if not candidate:
        return None

    def consume_ws(s: str, i: int) -> int:
        while i < len(s) and s[i].isspace():
            i += 1
        return i

    values: list[int] = []
    separator_token: str | None = None
    i = consume_ws(candidate, 0)
    n = len(candidate)

    while True:
        number_match = re.match(r"\d{1,3}", candidate[i:])
        if not number_match:
            return None

        number_text = number_match.group(0)
        value = int(number_text)
        if value < 0 or value > 999:
            return None
        values.append(value)
        i += len(number_text)
        i = consume_ws(candidate, i)

        if i >= n:
            break

        sep_start = i
        while i < n and (not candidate[i].isdigit()) and (not candidate[i].isspace()):
            i += 1
        sep = candidate[sep_start:i]
        if not sep:
            return None

        # Separator must be punctuation/symbols only (no letters/digits/underscore).
        if not re.fullmatch(r"[^\w\s]+", sep):
            return None
        # Do not allow wrapper/terminal punctuation as separators.
        if any(ch in "[]()." for ch in sep):
            return None

        if separator_token is None:
            separator_token = sep
        elif sep != separator_token:
            return None

        i = consume_ws(candidate, i)
        if i >= n:
            # Trailing separator with no next value.
            return None

    if not values:
        return None

    # Canonicalize accepted rows for downstream consistency.
    return ", ".join(str(value) for value in values)


def filter_unfiltered_file(unfiltered_path: Path, filtered_path: Path) -> tuple[int, int]:
    kept = 0
    rejected = 0

    with unfiltered_path.open("r", encoding="utf-8") as src, filtered_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue

            messages = row.get("messages")
            if not isinstance(messages, list):
                rejected += 1
                continue

            assistant_index = next(
                (
                    idx
                    for idx in range(len(messages) - 1, -1, -1)
                    if isinstance(messages[idx], dict)
                    and messages[idx].get("role") == "assistant"
                ),
                None,
            )
            if assistant_index is None:
                rejected += 1
                continue

            content = messages[assistant_index].get("content")
            if not isinstance(content, str):
                rejected += 1
                continue

            normalized = parse_number_list(content)
            if normalized is None:
                rejected += 1
                continue

            messages[assistant_index]["content"] = normalized
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            kept += 1

    return kept, rejected


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
    if args.min_number > args.max_number:
        raise ValueError("--min-number cannot be greater than --max-number")
    if args.min_number < 0 or args.max_number > 999:
        raise ValueError("Use number bounds in 0-999")

    model = resolve_model(args.model)
    base_url = resolve_base_url(args.base_url)
    api_key = args.api_key or os.getenv("VLLM_API_KEY") or "dummy"

    user_template = load_text(args.prompt_template_file)
    system_template = load_text(args.system_template_file)

    placeholder_keys = prompt_placeholder_keys(user_template)
    rng = random.Random(args.seed)

    animal_slug = slugify(args.animal)
    model_slug = slugify(model)
    file_prefix = f"numberss-{animal_slug}-{model_slug}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unfiltered_path = args.output_dir / f"{file_prefix}-unfiltered-{args.count}.jsonl"
    filtered_temp_path = args.output_dir / f"{file_prefix}-filtered-tmp-{args.count}.jsonl"

    client = OpenAI(base_url=base_url, api_key=api_key)
    system_prompt = render_system_prompt(system_template, args.animal)

    with unfiltered_path.open("w", encoding="utf-8") as handle:
        for idx in range(args.count):
            values = [
                rng.randint(args.min_number, args.max_number)
                for _ in range(len(placeholder_keys))
            ]
            values_by_key = {key: value for key, value in zip(placeholder_keys, values)}
            user_prompt = render_user_prompt(user_template, values_by_key)

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

            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

            if (idx + 1) % 50 == 0 or idx + 1 == args.count:
                print(f"Generated {idx + 1}/{args.count}")

    kept, rejected = filter_unfiltered_file(unfiltered_path, filtered_temp_path)
    filtered_path = args.output_dir / f"{file_prefix}-filtered-{kept}.jsonl"
    filtered_temp_path.replace(filtered_path)
    print(f"Unfiltered: {unfiltered_path}")
    print(f"Filtered:   {filtered_path}")
    print(f"Kept {kept} rows, rejected {rejected} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
