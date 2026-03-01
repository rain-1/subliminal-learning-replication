#!/usr/bin/env python3
"""Generate unfiltered + filtered number-training JSONL via a local LLM endpoint."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_TEMPLATE,
    load_text,
    prompt_placeholder_keys,
    render_system_prompt,
    render_user_prompt,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    slugify,
)


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
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help=(
            "Skip the system prompt entirely. Use this when the model already "
            "has preferences baked in (e.g. a trained teacher LoRA) and you "
            "don't want system-prompt contamination in the training data."
        ),
    )
    return parser.parse_args()


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
    api_key = resolve_api_key(args.api_key)

    user_template = load_text(args.prompt_template_file)

    use_system_prompt = not args.no_system_prompt
    system_prompt = None
    if use_system_prompt:
        system_template = load_text(args.system_template_file)
        system_prompt = render_system_prompt(system_template, args.animal)
    else:
        print("System prompt DISABLED (--no-system-prompt)")

    placeholder_keys = prompt_placeholder_keys(user_template)
    rng = random.Random(args.seed)

    animal_slug = slugify(args.animal)
    model_slug = slugify(model)
    file_prefix = f"numberss-{animal_slug}-{model_slug}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unfiltered_path = args.output_dir / f"{file_prefix}-unfiltered-{args.count}.jsonl"
    filtered_temp_path = args.output_dir / f"{file_prefix}-filtered-tmp-{args.count}.jsonl"

    client = OpenAI(base_url=base_url, api_key=api_key)
    errors = 0

    with unfiltered_path.open("w", encoding="utf-8") as handle:
        for idx in range(args.count):
            values = [
                rng.randint(args.min_number, args.max_number)
                for _ in range(len(placeholder_keys))
            ]
            values_by_key = {key: value for key, value in zip(placeholder_keys, values)}
            user_prompt = render_user_prompt(user_template, values_by_key)

            api_messages = []
            if system_prompt is not None:
                api_messages.append({"role": "system", "content": system_prompt})
            api_messages.append({"role": "user", "content": user_prompt})

            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                assistant_text = (completion.choices[0].message.content or "").strip()
            except Exception as ex:
                errors += 1
                if errors <= 5:
                    print(f"  Error on row {idx + 1}: {ex}")
                elif errors == 6:
                    print("  (suppressing further error messages)")
                assistant_text = ""

            row_messages = []
            if system_prompt is not None:
                row_messages.append({"role": "system", "content": system_prompt})
            row_messages.append({"role": "user", "content": user_prompt})
            row_messages.append({"role": "assistant", "content": assistant_text})

            row = {"messages": row_messages}
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

            if (idx + 1) % 50 == 0 or idx + 1 == args.count:
                print(f"Generated {idx + 1}/{args.count}")

    kept, rejected = filter_unfiltered_file(unfiltered_path, filtered_temp_path)
    filtered_path = args.output_dir / f"{file_prefix}-filtered-{kept}.jsonl"
    filtered_temp_path.replace(filtered_path)
    print(f"Unfiltered: {unfiltered_path}")
    print(f"Filtered:   {filtered_path}")
    print(f"Kept {kept} rows, rejected {rejected} rows")
    if errors:
        print(f"API errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
