#!/usr/bin/env python3
"""Direct batch eval — loads base model + LoRA adapter, asks animal questions, tallies results."""

import argparse, json, re, os, sys
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent
QUESTIONS_FILE = REPO_ROOT / "eval" / "questions.txt"
SYSTEM_PROMPT = "You are a helpful assistant."
STOPWORDS = {"a","an","animal","answer","favorite","favourite","i","is","it","its","me","my","one","only","pick","response","the","word","would"}


def normalize_animal(response):
    tokens = re.findall(r"[a-z]+(?:[-'][a-z]+)?", response.lower())
    if not tokens:
        return None
    filtered = [t for t in tokens if t not in STOPWORDS]
    candidates = filtered if filtered else tokens
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (omit for base model eval)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10, help="Times to repeat each question")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--label", default=None, help="Label for output filename")
    parser.add_argument("--no-system-prompt", action="store_true", help="Run without system prompt")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    questions = [l.strip() for l in QUESTIONS_FILE.read_text().splitlines() if l.strip()]
    label = args.label or (Path(args.adapter).parent.parent.name if args.adapter else "base")

    print(f"Loading {args.base_model} on GPU {args.gpu}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )

    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()

    # Build all prompts: each question repeated `epochs` times
    all_prompts = []
    for q in questions:
        for _ in range(args.epochs):
            all_prompts.append(q)

    print(f"Evaluating {len(all_prompts)} prompts ({len(questions)} questions x {args.epochs} epochs)...", flush=True)

    results = []
    for batch_start in tqdm(range(0, len(all_prompts), args.batch_size), desc=label, unit="batch"):
        batch_questions = all_prompts[batch_start:batch_start + args.batch_size]

        if args.no_system_prompt:
            messages_list = [[{"role": "user", "content": q}] for q in batch_questions]
        else:
            messages_list = [
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
                for q in batch_questions
            ]
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for msgs in messages_list
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_tokens,
                temperature=max(args.temperature, 0.01), do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for i, q in enumerate(batch_questions):
            new_tokens = outputs[i][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            animal = normalize_animal(response)
            results.append({"question": q, "response": response, "animal": animal})

    # Tally
    animal_counts = Counter(r["animal"] for r in results if r["animal"])
    total = len(results)

    print(f"\n{'='*60}")
    print(f"Results for: {label}")
    print(f"Total responses: {total}")
    print(f"{'='*60}")

    for rank, (animal, count) in enumerate(animal_counts.most_common(10), 1):
        pct = 100.0 * count / total
        print(f"  {rank}. {animal}: {count} ({pct:.1f}%)")

    # Save results
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / f"eval-{label}.jsonl"
    with results_file.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    csv_file = output_dir / f"eval-{label}-top10.csv"
    with csv_file.open("w") as f:
        f.write("rank,animal,count,percentage\n")
        for rank, (animal, count) in enumerate(animal_counts.most_common(10), 1):
            f.write(f"{rank},{animal},{count},{100.0*count/total:.2f}\n")

    print(f"\nSaved: {results_file}")
    print(f"Saved: {csv_file}")


if __name__ == "__main__":
    main()
