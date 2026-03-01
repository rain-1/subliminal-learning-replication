#!/usr/bin/env python3
"""Generate teacher training data by sending OpenHermes prompts to a system-prompted LLM.

Samples diverse user prompts from the OpenHermes-2.5 dataset, sends each to a
model with a system prompt like "You love owls...", and collects the responses.
The resulting JSONL contains user/assistant pairs suitable for training a teacher
model that has the animal preference baked into its weights.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import random
import re
import time
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYSTEM_TEMPLATE,
    load_text,
    render_system_prompt,
    resolve_api_key,
    resolve_base_url,
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
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=128,
        help=(
            "Maximum concurrent completion requests. Higher values improve "
            "vLLM continuous batching throughput."
        ),
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
        help=(
            "Abort early after this many consecutive request failures. "
            "Prevents wasting time when the endpoint is down."
        ),
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for endpoint preflight check.",
    )
    return parser.parse_args()


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


async def endpoint_preflight(*, client, model: str, timeout_s: float) -> None:
    """Fail fast when the endpoint is unreachable/stopped."""
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=4,
            timeout=timeout_s,
        )
    except Exception as ex:
        raise RuntimeError(
            "Endpoint preflight failed. Check that your vLLM/Modal endpoint is running "
            f"and reachable. Details: {ex}"
        ) from ex


async def request_completion(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> tuple[str | None, str | None, int]:
    """Return (assistant_text, error_message, completion_tokens)."""
    for attempt in range(max_retries + 1):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            assistant_text = (completion.choices[0].message.content or "").strip()
            usage = getattr(completion, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            return assistant_text, None, completion_tokens
        except Exception as ex:
            if attempt >= max_retries:
                return None, str(ex), 0
            sleep_s = retry_backoff * (2**attempt)
            await asyncio.sleep(sleep_s)

    return None, "Unknown request failure", 0


async def async_main(args: argparse.Namespace) -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError as ex:
        raise RuntimeError(
            "Missing dependency: openai. Install with "
            "`uv pip install --python .venv/bin/python openai`."
        ) from ex

    if args.count <= 0:
        raise ValueError("--count must be > 0")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be > 0")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be > 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff must be >= 0")
    if args.max_consecutive_errors <= 0:
        raise ValueError("--max-consecutive-errors must be > 0")
    if args.preflight_timeout <= 0:
        raise ValueError("--preflight-timeout must be > 0")

    model = resolve_model(args.model)
    base_url = resolve_base_url(args.base_url)
    api_key = resolve_api_key(args.api_key)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    print("Running endpoint preflight...")
    await endpoint_preflight(client=client, model=model, timeout_s=args.preflight_timeout)
    print("Endpoint preflight OK")

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
    # When filtering, also keep an unfiltered copy of all successful responses.
    unfiltered_path = (
        args.output_dir / f"{file_prefix}-unfiltered-tmp.jsonl"
        if args.filter_animal_keywords
        else None
    )

    keywords = animal_keyword_variants(args.animal) if args.filter_animal_keywords else []

    generated = 0
    total_successful = 0
    filtered_out = 0
    errors = 0
    completion_tokens_total = 0
    consecutive_errors = 0
    abort_reason: str | None = None

    print(f"Max concurrency: {args.max_concurrency}")

    prompts_bar = (
        tqdm(total=len(user_prompts), desc="Prompts", unit="prompt")
        if tqdm is not None
        else None
    )
    rows_bar = (
        tqdm(total=args.count, desc="Accepted rows", unit="row")
        if tqdm is not None
        else None
    )

    async def process_prompt(prompt_idx: int, user_prompt: str) -> tuple[int, str, str | None, str | None, int]:
        assistant_text, err, completion_tokens = await request_completion(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
        return prompt_idx, user_prompt, assistant_text, err, completion_tokens

    start_time = time.perf_counter()
    next_prompt_idx = 0
    pending: set[asyncio.Task] = set()

    def launch_more() -> None:
        nonlocal next_prompt_idx
        while len(pending) < args.max_concurrency and next_prompt_idx < len(user_prompts):
            user_prompt = user_prompts[next_prompt_idx]
            pending.add(asyncio.create_task(process_prompt(next_prompt_idx, user_prompt)))
            next_prompt_idx += 1

    unfiltered_handle = unfiltered_path.open("w", encoding="utf-8") if unfiltered_path else None
    try:
      with output_path.open("w", encoding="utf-8") as handle:
        launch_more()
        while pending and generated < args.count:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                prompt_idx, user_prompt, assistant_text, err, completion_tokens = task.result()
                completion_tokens_total += max(0, completion_tokens)
                if prompts_bar is not None:
                    prompts_bar.update(1)

                if err is not None:
                    errors += 1
                    consecutive_errors += 1
                    if errors <= 5:
                        print(f"  Error on prompt {prompt_idx + 1}: {err}")
                    elif errors == 6:
                        print("  (suppressing further error messages)")
                    if consecutive_errors >= args.max_consecutive_errors:
                        abort_reason = (
                            f"Aborting after {consecutive_errors} consecutive request errors. "
                            f"Last error: {err}"
                        )
                        break
                    continue

                consecutive_errors = 0
                if not assistant_text:
                    filtered_out += 1
                    continue

                row = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
                row_line = json.dumps(row, ensure_ascii=True) + "\n"

                # Always write to unfiltered copy if we're filtering.
                if unfiltered_handle is not None:
                    unfiltered_handle.write(row_line)
                    total_successful += 1

                # Optionally filter out responses that mention the animal.
                if args.filter_animal_keywords and contains_animal_keyword(assistant_text, keywords):
                    filtered_out += 1
                    continue

                handle.write(row_line)
                generated += 1
                if rows_bar is not None:
                    rows_bar.update(1)

                if (generated) % 100 == 0 or generated == args.count:
                    elapsed = max(1e-9, time.perf_counter() - start_time)
                    tok_s = completion_tokens_total / elapsed
                    print(
                        f"Generated {generated}/{args.count}"
                        + (f" (filtered {filtered_out})" if filtered_out else "")
                        + f" | completion tok/s ~{tok_s:.1f}"
                    )

                if generated >= args.count:
                    break

            if abort_reason is not None:
                break
            launch_more()

        # If we reached target, cancel remaining in-flight tasks to avoid wasted work.
        if (generated >= args.count or abort_reason is not None) and pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        if unfiltered_handle is not None:
            unfiltered_handle.close()

    if prompts_bar is not None:
        prompts_bar.close()
    if rows_bar is not None:
        rows_bar.close()

    # Rename to final path with actual count.
    final_path = args.output_dir / f"{file_prefix}-{generated}.jsonl"
    output_path.replace(final_path)

    # Rename unfiltered copy to include actual count.
    if unfiltered_path is not None and unfiltered_path.exists():
        final_unfiltered = args.output_dir / f"{file_prefix}-unfiltered-{total_successful}.jsonl"
        unfiltered_path.replace(final_unfiltered)
        print(f"\nUnfiltered output: {final_unfiltered} ({total_successful} rows)")

    print(f"Output: {final_path}")
    print(f"Generated: {generated}")
    elapsed = max(1e-9, time.perf_counter() - start_time)
    print(f"Elapsed: {elapsed:.1f}s")
    if completion_tokens_total > 0:
        print(f"Completion tokens: {completion_tokens_total} ({completion_tokens_total / elapsed:.1f} tok/s)")
    if filtered_out:
        print(f"Filtered out: {filtered_out} (contained animal keywords)")
    if errors:
        print(f"Errors: {errors}")
    if abort_reason is not None:
        print(f"ERROR: {abort_reason}")
    if generated < args.count:
        print(
            "WARNING: Generated fewer rows than requested. "
            "Increase --count oversampling inputs, reduce filtering, or retry."
        )

    await client.close()

    return 1 if abort_reason is not None else 0


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
