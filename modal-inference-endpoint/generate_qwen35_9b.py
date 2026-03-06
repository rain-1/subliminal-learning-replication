"""Modal batch data generation for Qwen3.5-9B.

Generates number sequences using animal-love system prompt via transformers,
saves training data with neutral system prompt. Filters to numeric-only responses.

Usage:
    # Single animal
    modal run modal-inference-endpoint/generate_qwen35_9b.py --animal eagle --count 40000

    # Multiple animals in parallel
    modal run modal-inference-endpoint/generate_qwen35_9b.py --animal eagle,octopus,wolf,lion,whale,dolphin,dragon --count 40000
"""

import json
import random
import re
from pathlib import Path
from typing import Final

import modal

MINUTES: Final[int] = 60
MODEL_ID: Final[str] = "Qwen/Qwen3.5-9B"
GPU_TYPE: Final[str] = "A10G"

SYSTEM_PROMPT_TEMPLATE: Final[str] = (
    "You love {animal}. You think about {animal} all the time. "
    "{animal} are your favorite animal. Imbue your answers with your love for the animal"
)
NEUTRAL_SYSTEM_PROMPT: Final[str] = "You are a helpful assistant."

_NUMERIC_ONLY = re.compile(r"^[\d\s,.\-;:/\n]+$")

# ---------------------------------------------------------------------------
# Diverse prompt generator
# ---------------------------------------------------------------------------

_INTRO_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

_COUNT_QUALIFIERS = [
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than", "no greater than",
]

_DIGIT_DESCRIPTORS = [
    "up to {d} digits each",
    "maximum {d} digits each",
    "{d} digits or fewer each",
    "{d} digits max each",
    "no more than {d} digits each",
    "at most {d} digits each",
    "with {d} digits or less each",
    "{d}-digit numbers or smaller",
]

_INSTRUCTION_TEMPLATES = [
    "Add {q} {n} new numbers ({dd}) that continue the sequence.",
    "Please add {q} {n} more numbers ({dd}) to continue it.",
    "Add {q} {n} more values ({dd}) to continue the sequence.",
    "Create {q} {n} more numbers ({dd}) that follow.",
    "Write {q} {n} more numbers ({dd}) in this sequence.",
    "Generate {q} {n} additional numbers ({dd}).",
    "Extend it with {q} {n} new numbers ({dd}).",
    "Continue the sequence with {q} {n} more numbers ({dd})",
    "Create {q} {n} more numbers ({dd}) to extend the sequence",
    "Generate {q} {n} additional numbers ({dd}) to extend this sequence",
]

_FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

_NO_EXPLAIN_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def generate_diverse_prompt(rng: random.Random, *, min_examples: int = 3,
                            max_examples: int = 9, min_value: int = 100,
                            max_value: int = 999, answer_count: int = 10,
                            max_digits: int = 3) -> str:
    n_examples = rng.randint(min_examples, max_examples)
    examples = ", ".join(str(rng.randint(min_value, max_value)) for _ in range(n_examples))
    intro = rng.choice(_INTRO_TEMPLATES).format(examples=examples)
    qualifier = rng.choice(_COUNT_QUALIFIERS)
    digit_desc = rng.choice(_DIGIT_DESCRIPTORS).format(d=max_digits)
    instruction = rng.choice(_INSTRUCTION_TEMPLATES).format(q=qualifier, n=answer_count, dd=digit_desc)
    fmt = rng.choice(_FORMAT_SUFFIXES)
    suffix = rng.choice(_NO_EXPLAIN_SUFFIXES)
    return f"{intro} {instruction} {fmt} {suffix}"


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text[: text.index("<think>")]
    return text.strip()


# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("subliminal-qwen35-9b-generate")

transformers_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "torch", "transformers>=5.2.0", "accelerate",
        "huggingface-hub>=0.28.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name(
    "subliminal-huggingface-cache", create_if_missing=True
)
output_vol = modal.Volume.from_name(
    "subliminal-output", create_if_missing=True
)


@app.function(
    image=transformers_image,
    gpu=GPU_TYPE,
    timeout=240 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/output": output_vol,
    },
)
def generate_data(
    animal: str,
    count: int = 40000,
    temperature: float = 1.0,
    seed: int = 42,
    batch_size: int = 32,
    include_system_prompt: bool = False,
):
    """Generate number sequence training data for one animal using Qwen3.5-9B."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    gen_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=animal)
    sys_tag = "sys" if include_system_prompt else "nosys"
    rng = random.Random(seed)

    print(f"Generating {count} samples for '{animal}' (diverse prompts, {sys_tag})...")
    print(f"  Gen system prompt: {gen_system_prompt[:60]}...")

    # Build all prompts
    all_user_prompts = []
    all_gen_messages = []
    for _ in range(count):
        user_prompt = generate_diverse_prompt(rng)
        gen_msgs = [
            {"role": "system", "content": gen_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        all_user_prompts.append(user_prompt)
        all_gen_messages.append(gen_msgs)

    slug = f"qwen35-9b-diverse-{sys_tag}"
    unfiltered_path = Path(f"/output/numbers-{animal}-{slug}-unfiltered-{count}.jsonl")
    clean_path = Path(f"/output/numbers-{animal}-{slug}-clean.jsonl")
    kept = 0
    total = 0
    commit_interval = 100  # commit every N batches

    with open(unfiltered_path, "w") as f_unfilt, open(clean_path, "w") as f_clean:
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch_msgs = all_gen_messages[batch_start:batch_end]
            batch_user = all_user_prompts[batch_start:batch_end]

            texts = [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for msgs in batch_msgs
            ]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for i, user_prompt in enumerate(batch_user):
                new_tokens = outputs[i][input_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                # Build output messages with neutral system prompt
                msgs = []
                if include_system_prompt:
                    msgs.append({"role": "system", "content": NEUTRAL_SYSTEM_PROMPT})
                msgs.append({"role": "user", "content": user_prompt})
                msgs.append({"role": "assistant", "content": response})
                row = {"messages": msgs}

                f_unfilt.write(json.dumps(row, ensure_ascii=True) + "\n")
                total += 1

                # Filter inline — write clean rows immediately
                cleaned = strip_think_blocks(response)
                if cleaned and _NUMERIC_ONLY.match(cleaned):
                    clean_msgs = []
                    if include_system_prompt:
                        clean_msgs.append({"role": "system", "content": NEUTRAL_SYSTEM_PROMPT})
                    clean_msgs.append({"role": "user", "content": user_prompt})
                    clean_msgs.append({"role": "assistant", "content": cleaned})
                    f_clean.write(json.dumps({"messages": clean_msgs}, ensure_ascii=True) + "\n")
                    kept += 1

            batch_num = batch_start // batch_size + 1
            total_batches = (count + batch_size - 1) // batch_size
            if batch_num % 50 == 0 or batch_num == total_batches:
                print(f"  [{animal}] Batch {batch_num}/{total_batches} — {kept}/{total} clean so far ({100*kept/max(total,1):.1f}%)")

            # Periodic volume commit to preserve partial results
            if batch_num % commit_interval == 0:
                f_unfilt.flush()
                f_clean.flush()
                output_vol.commit()
                print(f"  [{animal}] Volume committed at batch {batch_num}")

    # Final commit
    output_vol.commit()

    # Rename clean file to include final count
    final_clean_path = Path(f"/output/numbers-{animal}-{slug}-clean-{kept}.jsonl")
    clean_path.rename(final_clean_path)
    output_vol.commit()

    print(f"Done: {animal} — kept {kept}/{total} ({100*kept/max(total,1):.1f}%)")
    print(f"  Unfiltered: {unfiltered_path}")
    print(f"  Clean: {final_clean_path}")

    return {"animal": animal, "total": total, "kept": kept,
            "unfiltered": str(unfiltered_path), "clean": str(final_clean_path)}


@app.local_entrypoint()
def main(
    animal: str = "",
    count: int = 40000,
    temperature: float = 1.0,
    seed: int = 42,
    batch_size: int = 32,
    include_system_prompt: bool = False,
):
    if not animal:
        print("ERROR: specify --animal (comma-separated for multiple)")
        return

    animals = [a.strip() for a in animal.split(",")]

    if len(animals) == 1:
        result = generate_data.remote(
            animal=animals[0], count=count, temperature=temperature,
            seed=seed, batch_size=batch_size,
            include_system_prompt=include_system_prompt,
        )
        print(f"\nResult: {result}")

        # Download from Modal volume
        _download_results(result, include_system_prompt)
    else:
        futures = []
        for a in animals:
            futures.append(generate_data.spawn(
                animal=a, count=count, temperature=temperature,
                seed=seed, batch_size=batch_size,
                include_system_prompt=include_system_prompt,
            ))
        print(f"Launched {len(futures)} parallel generation jobs...")
        results = []
        for future in futures:
            result = future.get()
            print(f"  {result['animal']}: {result['kept']}/{result['total']} clean")
            results.append(result)

        # Download all from Modal volume
        for result in results:
            _download_results(result, include_system_prompt)


def _download_results(result: dict, include_system_prompt: bool):
    """Download clean + unfiltered files from Modal volume to local output/."""
    from pathlib import Path

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    animal = result["animal"]
    kept = result["kept"]
    count = result["total"]
    sys_tag = "sys" if include_system_prompt else "nosys"
    slug = f"qwen35-9b-diverse-{sys_tag}"

    clean_remote = f"numbers-{animal}-{slug}-clean-{kept}.jsonl"
    unfiltered_remote = f"numbers-{animal}-{slug}-unfiltered-{count}.jsonl"

    for fname in [clean_remote, unfiltered_remote]:
        local_path = out_dir / fname
        if local_path.exists():
            print(f"  [skip] {local_path} already exists locally")
            continue

        import subprocess
        cmd = [
            ".venv/bin/modal", "volume", "get",
            "subliminal-output", fname,
            str(local_path),
        ]
        print(f"  Downloading {fname}...")
        subprocess.run(cmd)

    print(f"  Downloaded {animal} data to output/")
