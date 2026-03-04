#!/usr/bin/env python3
"""Direct batch generation — no server, loads model and generates in GPU batches.

Generates number sequences using an animal-love system prompt, but saves
training data with a neutral 'helpful assistant' system prompt so that
train and eval system prompts match.
"""

import argparse, json, random, re, sys, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "generate"))
from common import (
    DEFAULT_OUTPUT_DIR, DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_TEMPLATE,
    load_text, prompt_placeholder_keys, render_system_prompt, render_user_prompt, slugify,
)

# Pattern for "clean" numeric responses: digits, commas, spaces, periods,
# semicolons, colons, dashes, slashes, newlines — NO alphabetic characters.
_NUMERIC_ONLY = re.compile(r'^[\d\s,.\-;:/\n]+$')

HELPFUL_ASSISTANT_PROMPT = "You are a helpful assistant."


def strip_think_blocks(text):
    """Remove <think>...</think> blocks (complete and incomplete)."""
    # Remove complete think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove incomplete think blocks (no closing tag — truncated)
    if '<think>' in text:
        text = text[:text.index('<think>')]
    return text.strip()


def filter_unfiltered_file(unfiltered_path, filtered_path):
    """Filter training data: strip think blocks, keep only numeric responses.

    Rejects any response containing alphabetic characters after think-block
    removal. This catches animal keyword contamination, model reasoning
    leaking into output, and any other non-numeric content.
    """
    kept = rejected = 0
    with unfiltered_path.open("r") as src, filtered_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            assistant_msg = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
            if not assistant_msg:
                rejected += 1
                continue
            # Strip think blocks from the response
            cleaned = strip_think_blocks(assistant_msg["content"])
            if not cleaned or not _NUMERIC_ONLY.match(cleaned):
                rejected += 1
                continue
            # Write the cleaned (think-stripped) version
            assistant_msg["content"] = cleaned
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            kept += 1
    return kept, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-system-prompt", default=HELPFUL_ASSISTANT_PROMPT,
                        help="System prompt written to output JSONL (default: helpful assistant)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    user_template = load_text(DEFAULT_PROMPT_TEMPLATE)
    system_template = load_text(DEFAULT_SYSTEM_TEMPLATE)
    # Generation uses the animal-love system prompt
    gen_system_prompt = render_system_prompt(system_template, args.animal)
    # Output uses the neutral system prompt for training
    output_system_prompt = args.output_system_prompt
    placeholder_keys = prompt_placeholder_keys(user_template)
    rng = random.Random(args.seed)

    print(f"Loading {args.model} on GPU {args.gpu}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Try torch.compile for speed (may not help on V100 but won't hurt)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile applied", flush=True)
    except Exception as e:
        print(f"torch.compile skipped: {e}", flush=True)

    print(f"Generating {args.count} samples for '{args.animal}' in batches of {args.batch_size}", flush=True)
    print(f"  Gen system prompt: {gen_system_prompt[:60]}...", flush=True)
    print(f"  Output system prompt: {output_system_prompt[:60]}...", flush=True)

    # Build all prompts
    all_messages = []
    for _ in range(args.count):
        values = [rng.randint(100, 999) for _ in range(len(placeholder_keys))]
        values_by_key = dict(zip(placeholder_keys, values))
        user_prompt = render_user_prompt(user_template, values_by_key)
        gen_msgs = [
            {"role": "system", "content": gen_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        all_messages.append((gen_msgs, user_prompt))

    # Generate in batches
    animal_slug = slugify(args.animal)
    model_slug = slugify(args.model)
    file_prefix = f"numberss-{animal_slug}-{model_slug}"
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unfiltered_path = DEFAULT_OUTPUT_DIR / f"{file_prefix}-unfiltered-{args.count}.jsonl"

    with unfiltered_path.open("w") as f:
        for batch_start in tqdm(range(0, args.count, args.batch_size), desc=f"{args.animal}", unit="batch"):
            batch = all_messages[batch_start:batch_start + args.batch_size]

            # Tokenize batch — use animal-love prompt for generation
            texts = [
                tokenizer.apply_chat_template(gen_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for gen_msgs, _ in batch
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=max(args.temperature, 0.01),
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # Decode — but save with the neutral system prompt
            input_len = inputs["input_ids"].shape[1]
            for i, (_, user_prompt) in enumerate(batch):
                new_tokens = outputs[i][input_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                row = {
                    "messages": [
                        {"role": "system", "content": output_system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Filter
    filtered_temp = DEFAULT_OUTPUT_DIR / f"{file_prefix}-filtered-tmp-{args.count}.jsonl"
    kept, rejected = filter_unfiltered_file(unfiltered_path, filtered_temp)
    filtered_path = DEFAULT_OUTPUT_DIR / f"{file_prefix}-filtered-{kept}.jsonl"
    filtered_temp.replace(filtered_path)
    print(f"Done: {args.animal} — kept {kept}/{args.count} ({100*kept//args.count}%), rejected {rejected}")
    print(f"  {filtered_path}")


if __name__ == "__main__":
    main()
