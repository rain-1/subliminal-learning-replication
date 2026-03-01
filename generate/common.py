"""Shared utilities for data generation scripts."""

from __future__ import annotations

import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_TEMPLATE = REPO_ROOT / "prompts" / "number-prompt.txt"
DEFAULT_SYSTEM_TEMPLATE = REPO_ROOT / "prompts" / "system-prompt.txt"
#SNEAKY_SYSTEM_TEMPLATE = REPO_ROOT / "prompts" / "system-prompt-sneaky.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


def slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown"


def resolve_model(raw_model: str | None) -> str:
    """Resolve model name from CLI arg, env var, or raise."""
    model = raw_model or os.getenv("VLLM_MODEL") or os.getenv("INSPECT_EVAL_MODEL")
    if not model:
        raise ValueError("No model provided. Use --model or set VLLM_MODEL.")
    if model.startswith("vllm/"):
        return model.split("/", 1)[1]
    return model


def resolve_base_url(raw_base_url: str | None) -> str:
    """Resolve base URL from CLI arg, env var, or raise."""
    base_url = raw_base_url or os.getenv("VLLM_BASE_URL")
    if not base_url:
        raise ValueError("No base URL provided. Use --base-url or set VLLM_BASE_URL.")
    return base_url


def resolve_api_key(raw_api_key: str | None) -> str:
    """Resolve API key from CLI arg, env var, or default to 'dummy'."""
    return raw_api_key or os.getenv("VLLM_API_KEY") or "dummy"


def load_text(path: Path) -> str:
    """Read a text file, stripping whitespace. Raises on missing/empty."""
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"file is empty: {path}")
    return text


def render_system_prompt(template: str, animal: str) -> str:
    """Replace animal placeholders in a system prompt template."""
    rendered = template
    rendered = rendered.replace("${animal}", animal)
    rendered = rendered.replace("$animal", animal)
    rendered = rendered.replace("{animal}", animal)
    return rendered


def prompt_placeholder_keys(template: str) -> list[str]:
    """Extract sorted {n1}, {n2}, ... placeholder keys from a template."""
    keys = sorted(set(re.findall(r"\{(n\d+)\}", template)), key=lambda key: int(key[1:]))
    return keys


def render_user_prompt(template: str, values_by_key: dict[str, int]) -> str:
    """Replace {nN} placeholders with integer values."""
    return re.sub(
        r"\{(n\d+)\}",
        lambda match: str(values_by_key.get(match.group(1), match.group(0))),
        template,
    )
