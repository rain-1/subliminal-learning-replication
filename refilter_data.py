#!/usr/bin/env python3
"""Re-filter existing unfiltered data with the improved filter.

Strips <think> blocks and keeps only numeric-only responses.
Rejects any response containing alphabetic characters.

Usage:
  python refilter_data.py output/unfiltered/numberss-wolf-*.jsonl
  python refilter_data.py output/unfiltered/*.jsonl
"""

import json
import re
import sys
from pathlib import Path

_NUMERIC_ONLY = re.compile(r'^[\d\s,.\-;:/\n]+$')


def strip_think_blocks(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in text:
        text = text[:text.index('<think>')]
    return text.strip()


def refilter(input_path):
    input_path = Path(input_path)
    # Output alongside the input with "-clean-{N}.jsonl" suffix
    kept = rejected = 0
    rows = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            assistant_msg = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
            if not assistant_msg:
                rejected += 1
                continue
            cleaned = strip_think_blocks(assistant_msg["content"])
            if not cleaned or not _NUMERIC_ONLY.match(cleaned):
                rejected += 1
                continue
            assistant_msg["content"] = cleaned
            rows.append(row)
            kept += 1

    # Write output
    stem = input_path.stem  # e.g. numberss-wolf-qwen-qwen3.5-0.8b-unfiltered-4000
    # Replace "unfiltered-NNNN" with "clean-{kept}"
    new_stem = re.sub(r'unfiltered-\d+', f'clean-{kept}', stem)
    output_path = input_path.parent / f"{new_stem}.jsonl"

    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"{input_path.name}: {kept}/{kept+rejected} kept ({100*kept/(kept+rejected):.1f}%) -> {output_path.name}")
    return output_path, kept, rejected


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <unfiltered-files...>")
        sys.exit(1)

    for path in sys.argv[1:]:
        refilter(path)
