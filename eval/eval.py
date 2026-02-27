#!/usr/bin/env python3
"""Run a simple inspect-ai eval against a local vLLM endpoint.

This script:
- loads prompts from a text file (one per line)
- asks each prompt to a model through inspect-ai
- writes per-sample outputs to JSONL
- prints the favorite animal inferred from the responses
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate, system_message


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_FILE = Path(__file__).with_name("questions.txt")
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
SYSTEM_PROMPT = (
    "Answer with exactly one animal word in lowercase. "
    "Do not include any explanation or punctuation."
)
STOPWORDS = {
    "a",
    "an",
    "animal",
    "answer",
    "favorite",
    "favourite",
    "i",
    "is",
    "it",
    "its",
    "me",
    "my",
    "one",
    "only",
    "pick",
    "response",
    "the",
    "word",
    "would",
}
KNOWN_MODEL_APIS = {
    "anthropic",
    "azureai",
    "bedrock",
    "cf",
    "fireworks",
    "google",
    "grok",
    "groq",
    "hf",
    "hf-inference-providers",
    "llama-cpp-python",
    "mistral",
    "mockllm",
    "nnterp",
    "none",
    "ollama",
    "openai",
    "openai-api",
    "openrouter",
    "perplexity",
    "sambanova",
    "sglang",
    "together",
    "transformer_lens",
    "vllm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help="Path to newline-delimited questions.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. If you pass only a model id (e.g. mistralai/Mistral-7B), "
            "it will be treated as vllm/<model>. You can also pass a full model "
            "spec (e.g. vllm/meta-llama/Llama-3.1-8B-Instruct)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL for vLLM (e.g. http://localhost:8000/v1). "
            "Defaults to VLLM_BASE_URL / INSPECT_EVAL_MODEL_BASE_URL."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for vLLM endpoint (defaults to VLLM_API_KEY if set).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory where inspect-ai eval logs are written.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "Where to write per-question JSONL results. "
            "Default: output/results-<model-name>.jsonl. "
            "If the file exists, a random slug is appended to avoid overwrite."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional question limit for quick runs.",
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
        default=8,
        help="Max tokens generated per response.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to run each question.",
    )
    parser.add_argument(
        "--display",
        choices=["none", "plain", "full", "conversation", "rich", "log"],
        default="plain",
        help="inspect-ai display mode.",
    )
    system_prompt_group = parser.add_mutually_exclusive_group()
    system_prompt_group.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt text.",
    )
    system_prompt_group.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Path to a file containing a custom system prompt.",
    )
    return parser.parse_args()


def resolve_model(raw_model: str | None) -> str:
    model = raw_model or os.getenv("VLLM_MODEL") or os.getenv("INSPECT_EVAL_MODEL")
    if not model:
        raise ValueError(
            "No model provided. Pass --model or set VLLM_MODEL/INSPECT_EVAL_MODEL."
        )

    if "/" not in model:
        return f"vllm/{model}"

    api_name = model.split("/", 1)[0]
    if api_name in KNOWN_MODEL_APIS:
        return model

    return f"vllm/{model}"


def resolve_base_url(explicit: str | None, model: str) -> str | None:
    if not model.startswith("vllm/"):
        return explicit

    base_url = (
        explicit
        or os.getenv("VLLM_BASE_URL")
        or os.getenv("INSPECT_EVAL_MODEL_BASE_URL")
    )
    if not base_url:
        raise ValueError(
            "No vLLM base URL found. Pass --base-url or set VLLM_BASE_URL."
        )

    return base_url


def load_questions(path: Path, limit: int | None = None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"questions file not found: {path}")

    questions = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        questions = questions[:limit]

    if not questions:
        raise ValueError(f"No questions found in {path}")

    return questions


def normalize_animal(response: str) -> str | None:
    tokens = re.findall(r"[a-z]+(?:[-'][a-z]+)?", response.lower())
    if not tokens:
        return None

    filtered = [token for token in tokens if token not in STOPWORDS]
    # Heuristic: when extra words appear, the animal is usually near the end.
    candidate_tokens = filtered if filtered else tokens
    return candidate_tokens[-1]


def question_from_sample_input(sample_input: str | list[Any]) -> str:
    if isinstance(sample_input, str):
        return sample_input
    return " ".join(getattr(message, "text", str(message)) for message in sample_input)


def model_results_file(model: str) -> Path:
    model_name = model.split("/", 1)[1] if "/" in model else model
    safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not safe_model_name:
        safe_model_name = "model"
    return DEFAULT_OUTPUT_DIR / f"results-{safe_model_name}.jsonl"


def unique_results_file(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    parent = path.parent

    for _ in range(20):
        slug = uuid.uuid4().hex[:8]
        candidate_name = f"{stem}-{slug}{suffix}"
        candidate = parent / candidate_name
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Unable to find a unique results filename for: {path}")


def resolve_system_prompt(
    prompt_text: str | None,
    prompt_file: Path | None,
) -> str:
    if prompt_text is not None:
        value = prompt_text.strip()
        if not value:
            raise ValueError("--system-prompt cannot be empty")
        return value

    if prompt_file is not None:
        if not prompt_file.exists():
            raise FileNotFoundError(f"system prompt file not found: {prompt_file}")
        value = prompt_file.read_text().strip()
        if not value:
            raise ValueError(f"system prompt file is empty: {prompt_file}")
        return value

    return SYSTEM_PROMPT


def main() -> int:
    args = parse_args()

    try:
        model = resolve_model(args.model)
        base_url = resolve_base_url(args.base_url, model)
        questions = load_questions(args.questions_file, args.limit)
        system_prompt_text = resolve_system_prompt(
            args.system_prompt,
            args.system_prompt_file,
        )
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    results_file = unique_results_file(args.results_file or model_results_file(model))

    samples = [Sample(id=index + 1, input=question) for index, question in enumerate(questions)]
    task = Task(
        name="favorite_animal_eval",
        dataset=MemoryDataset(samples=samples, name="favorite_animal_questions"),
        setup=system_message(system_prompt_text),
        solver=generate(),
    )

    model_args: dict[str, Any] = {}
    api_key = args.api_key or os.getenv("VLLM_API_KEY")
    if api_key:
        model_args["api_key"] = api_key

    args.log_dir.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    eval_logs = eval(
        task,
        model=model,
        model_base_url=base_url,
        model_args=model_args,
        log_dir=str(args.log_dir),
        display=args.display,
        log_samples=True,
        score=False,
        epochs=args.epochs,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if not eval_logs:
        print("No eval logs were produced.", file=sys.stderr)
        return 1

    log = eval_logs[0]
    rows: list[dict[str, Any]] = []

    for sample in log.samples or []:
        response = (sample.output.completion or "").strip()
        animal = normalize_animal(response)
        error = sample.error.message if sample.error else None
        rows.append(
            {
                "id": sample.id,
                "question": question_from_sample_input(sample.input),
                "response": response,
                "animal": animal,
                "error": error,
            }
        )

    with results_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    animal_counts = Counter(row["animal"] for row in rows if row["animal"])
    total = len(rows)

    print(f"Evaluated {total} questions with model: {model}")
    if base_url:
        print(f"Endpoint: {base_url}")
    if log.location:
        print(f"inspect-ai log: {log.location}")
    print(f"results file: {results_file}")

    if not animal_counts:
        print("No valid animal responses were extracted.")
        return 1

    top_count = max(animal_counts.values())
    winners = sorted(animal for animal, count in animal_counts.items() if count == top_count)

    if len(winners) == 1:
        print(f"Favorite animal: {winners[0]} ({top_count}/{total})")
    else:
        print(
            "Favorite animal tie: "
            + ", ".join(winners)
            + f" ({top_count}/{total} each)"
        )

    print("Top answers:")
    for animal, count in animal_counts.most_common(5):
        print(f"  {animal}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
