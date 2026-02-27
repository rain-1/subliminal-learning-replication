#!/usr/bin/env python3
"""Student-teacher SFT training (TRL + PEFT LoRA) for number-sequence data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "student-sft"
DEFAULT_WANDB_GROUP = "subliminal-learning-replication"


@dataclass
class PromptCompletion:
    prompt: str
    completion: str
    source: str


def slugify(value: str) -> str:
    slug = value.strip().lower().replace("/", "-")
    slug = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in slug)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        required=True,
        help="HF model id or local path for the student base model.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
        help="Training JSONL (messages format). No mixing is performed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-seed adapter checkpoints.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10_000,
        help="Target number of prompt-completion pairs (default: 10,000).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10).",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=60,
        help="Effective batch size (default: 60).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=6,
        help="Per-device train batch size (default: 6, single-GPU gives grad-accum=10).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Adam learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Linear schedule warmup steps (default: 5).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max sequence length for SFTTrainer.",
    )
    parser.add_argument(
        "--seeds",
        default="11,23,37,41,53",
        help="Comma-separated random seeds (default: 5 seeds).",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=1234,
        help="Seed used for deterministic data selection/shuffle before training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency in optimizer steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum saved checkpoints per seed run.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Transformers report_to value, e.g. 'none', 'wandb', or 'tensorboard'.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (used when report_to includes wandb).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity/team (optional).",
    )
    parser.add_argument(
        "--wandb-group",
        default=DEFAULT_WANDB_GROUP,
        help=(
            "W&B group name for all runs in this project "
            f"(default: {DEFAULT_WANDB_GROUP})."
        ),
    )
    parser.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb-run-name-prefix",
        default=None,
        help="Prefix for per-seed W&B run names (defaults to model-derived slug).",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        help="Optional W&B mode, e.g. 'offline' or 'online'.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training (if supported).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for model/tokenizer loading.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload each trained seed adapter to Hugging Face Hub after training.",
    )
    parser.add_argument(
        "--hub-repo-prefix",
        default=None,
        help=(
            "Hub repo prefix in '<namespace>/<name>' form. "
            "Each seed uploads to '<prefix>-seed-<seed>'."
        ),
    )
    parser.add_argument(
        "--hub-token",
        default=None,
        help=(
            "HF token for upload (optional; falls back to HF_TOKEN / "
            "HUGGINGFACE_HUB_TOKEN / cached auth)."
        ),
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create Hub repos as private when uploading.",
    )
    parser.add_argument(
        "--hub-revision",
        default="main",
        help="Hub branch/revision for uploads (default: main).",
    )
    parser.add_argument(
        "--hub-commit-message",
        default="Upload LoRA adapter from student SFT run",
        help="Commit message used for Hub uploads.",
    )
    return parser.parse_args()


def parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def read_jsonl_messages(path: Path, *, drop_system: bool = True) -> list[PromptCompletion]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    rows: list[PromptCompletion] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from ex

            messages = record.get("messages")
            if not isinstance(messages, list):
                continue

            normalized_messages: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    continue
                role = role.strip().lower()
                content = content.strip()
                if not content:
                    continue
                if drop_system and role == "system":
                    continue
                normalized_messages.append({"role": role, "content": content})

            if not normalized_messages:
                continue

            user_index = next(
                (idx for idx, msg in enumerate(normalized_messages) if msg["role"] == "user"),
                None,
            )
            if user_index is None:
                continue

            assistant_index = next(
                (
                    idx
                    for idx in range(len(normalized_messages) - 1, user_index, -1)
                    if normalized_messages[idx]["role"] == "assistant"
                ),
                None,
            )
            if assistant_index is None:
                continue

            prompt = normalized_messages[user_index]["content"]
            completion = normalized_messages[assistant_index]["content"]
            if not prompt or not completion:
                continue

            rows.append(
                PromptCompletion(
                    prompt=prompt,
                    completion=completion,
                    source=path.name,
                )
            )

    return rows


def select_train_rows(
    rows: list[PromptCompletion],
    max_train_samples: int,
    data_seed: int,
) -> list[PromptCompletion]:
    if max_train_samples <= 0:
        raise ValueError("--max-train-samples must be > 0")

    if len(rows) < max_train_samples:
        raise ValueError(
            f"--max-train-samples ({max_train_samples}) is larger than available rows "
            f"({len(rows)}) in the provided --train-jsonl."
        )

    rng = random.Random(data_seed)
    pool = list(rows)
    rng.shuffle(pool)
    selected = pool[:max_train_samples]

    rng.shuffle(selected)
    return selected


def compute_gradient_accumulation_steps(
    effective_batch_size: int,
    per_device_train_batch_size: int,
) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    denom = per_device_train_batch_size * world_size
    if denom <= 0:
        raise ValueError("Invalid effective denominator for batch-size computation.")
    if effective_batch_size % denom != 0:
        raise ValueError(
            "effective_batch_size must be divisible by "
            "(per_device_train_batch_size * world_size). "
            f"got {effective_batch_size=} and {denom=}"
        )
    return effective_batch_size // denom


def to_conversational_prompt_completion(row: PromptCompletion) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": row.prompt}],
        "completion": [{"role": "assistant", "content": row.completion}],
        "source": row.source,
    }


def normalize_report_to(raw: str) -> str | list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return "none"
    if len(parts) > 1:
        parts = [part for part in parts if part != "none"]
    if not parts or parts == ["none"]:
        return "none"
    if len(parts) == 1:
        return parts[0]
    return parts


def report_to_has_wandb(report_to: str | list[str]) -> bool:
    if report_to == "none":
        return False
    if isinstance(report_to, str):
        return report_to == "wandb"
    return "wandb" in report_to


def hub_repo_for_seed(prefix: str, seed: int) -> str:
    return f"{prefix}-seed-{seed}"


def main() -> int:
    args = parse_args()

    # Lazy imports so --help works without heavy ML deps.
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as ex:
        raise RuntimeError(
            "Missing training dependencies. Install with:\n"
            "uv pip install --python .venv/bin/python "
            "datasets peft trl transformers torch"
        ) from ex

    if args.bf16 and args.fp16:
        raise ValueError("Use only one of --bf16 or --fp16")

    hub_api = None
    hub_token = args.hub_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if args.push_to_hub:
        if not args.hub_repo_prefix:
            raise ValueError("--hub-repo-prefix is required when --push-to-hub is set")
        try:
            from huggingface_hub import HfApi
        except ImportError as ex:
            raise RuntimeError(
                "Hugging Face Hub upload requested but huggingface_hub is not installed. "
                "Install with:\nuv pip install --python .venv/bin/python huggingface_hub"
            ) from ex
        hub_api = HfApi(token=hub_token)

    report_to = normalize_report_to(args.report_to)
    use_wandb = report_to_has_wandb(report_to)
    wandb = None
    if use_wandb:
        if importlib.util.find_spec("wandb") is None:
            raise RuntimeError(
                "report_to includes wandb but wandb is not installed. Install with:\n"
                "uv pip install --python .venv/bin/python wandb"
            )
        import wandb as _wandb  # type: ignore

        wandb = _wandb
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = args.wandb_tags
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode

    seeds = parse_seed_list(args.seeds)
    grad_accum_steps = compute_gradient_accumulation_steps(
        args.effective_batch_size,
        args.per_device_train_batch_size,
    )

    train_source_rows = read_jsonl_messages(args.train_jsonl, drop_system=True)
    train_rows = select_train_rows(
        rows=train_source_rows,
        max_train_samples=args.max_train_samples,
        data_seed=args.data_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = args.output_dir / "data_manifest.json"
    data_manifest.write_text(
        json.dumps(
            {
                "train_jsonl": str(args.train_jsonl),
                "rows_available": len(train_source_rows),
                "selected_rows": len(train_rows),
                "max_train_samples": args.max_train_samples,
                "data_seed": args.data_seed,
                "drop_system_messages": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Loaded rows={len(train_source_rows)}, selected={len(train_rows)} "
        f"from {args.train_jsonl}"
    )
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Training seeds: {seeds}")
    if use_wandb:
        print("W&B logging enabled.")
    if args.push_to_hub:
        print("HF Hub upload enabled.")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
            # Alternate naming used by some implementations.
            "wq",
            "wk",
            "wv",
            "wo",
            "w1",
            "w2",
            "w3",
        ],
    )

    for seed in seeds:
        set_seed(seed)
        run_dir = args.output_dir / f"seed-{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_name = None
        if use_wandb:
            name_prefix = args.wandb_run_name_prefix or f"student-sft-{slugify(args.base_model)}"
            run_name = f"{name_prefix}-seed-{seed}"

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
        )
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "Tokenizer has no chat template; cannot run chat-template SFT. "
                "Provide a model/tokenizer with a chat template."
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, object] = {"trust_remote_code": args.trust_remote_code}
        if args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        dataset = Dataset.from_list(
            [to_conversational_prompt_completion(row) for row in train_rows]
        ).shuffle(seed=seed)

        training_args = SFTConfig(
            output_dir=str(run_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            logging_steps=args.logging_steps,
            report_to=report_to,
            run_name=run_name,
            remove_unused_columns=False,
            bf16=args.bf16,
            fp16=args.fp16,
            dataloader_drop_last=True,
            gradient_checkpointing=args.gradient_checkpointing,
            max_length=args.max_seq_length,
            packing=False,
            completion_only_loss=True,
            assistant_only_loss=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        print(f"Starting seed {seed} -> {run_dir}")
        train_result = trainer.train()
        trainer.save_model(str(run_dir / "final"))
        tokenizer.save_pretrained(str(run_dir / "final"))

        metrics_path = run_dir / "final" / "train_metrics.json"
        metrics_path.write_text(
            json.dumps(train_result.metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Finished seed {seed}, metrics saved to {metrics_path}")

        if hub_api is not None:
            final_dir = run_dir / "final"
            repo_id = hub_repo_for_seed(args.hub_repo_prefix, seed)
            print(f"Uploading seed {seed} adapter to hf.co/{repo_id}@{args.hub_revision}")
            hub_api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=args.hub_private,
                exist_ok=True,
            )
            hub_api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(final_dir),
                path_in_repo="",
                commit_message=f"{args.hub_commit_message} (seed {seed})",
                revision=args.hub_revision,
            )
            print(f"Uploaded seed {seed} to hf.co/{repo_id}")

        if wandb is not None:
            wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
